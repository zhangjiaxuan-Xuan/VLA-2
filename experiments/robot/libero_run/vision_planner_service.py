

"""
zsh:
conda activate /root/miniconda3/envs/ftdino   
python /zhaohan/ZJX/Agentic_VLA/code/agentic-robot/experiments/robot/libero_run/vision_planner_service.py \
  --endpoint ipc:///tmp/vision_planner.sock --device cuda:0 
  
"""

"""
VisionPlanner Service (GLM + mmgrounded-DINO + Qwen VLM)
- Persistently loaded:
    * GLM (initGLMT)
    * mmgrounded-DINO (init_detector)
    * Qwen VLM (load_qwen_vl_model)
- Provides 3 commands:
    1) plan:   text + initial frame -> (subtask plan, objs, locas)
    2) detect: initial frame object detection -> dirt  (the main process uses SAM/Cutie for local initialization and tracking)
    3) verify: uses a queue of recent frame pairs + the current subtask instruction -> bool (is the subtask complete), judged by Qwen

Communication: ZeroMQ REP over IPC (Unix domain socket) for minimal local overhead.
Message format: multipart
  - part[0] = utf-8 JSON, e.g. {"cmd":"plan", ...}
  - part[1:] = binary frames (image JPEG/PNG bytes or pickle bytes)
"""

import os
import sys
import io
import json
import time
import pickle
import traceback
from typing import Dict, Any, List, Tuple
from collections import deque

import zmq
import numpy as np
from PIL import Image
import torch

# ==== Project Path ====
ROOT = "path/to/your/VLA-2/.."
# sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "agentic-robot"))

# ==== Existing Functions ====
from script.auto_DL import initGLMT
from script.Wholebody import goforward
from script.mmgdino import mmgdino, init_detector
from qwenvl import load_qwen_vl_model, check_completion_with_qwen_vl
import signal
def handle_exit(signum, frame):
    print("[Service] get signal, exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)


# ---------- Image Decoding ----------
def _decode_image_from_bytes(b: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(b)) as im:
        return np.array(im.convert("RGB"))  # HWC uint8


def _encode_image_to_jpeg(arr: np.ndarray, quality: int = 85) -> bytes:
    assert arr.dtype == np.uint8 and arr.ndim == 3
    im = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ---------- Service ----------
class VisionPlannerService:
    """
    Persistently loaded resources:
        * GLM_pro, GLM_mod
        * mmgdino_model
        * Qwen (vlm_model, vlm_processor)
    """
    def __init__(self, device: str = "cuda:0", qwen_model_id: str = "path/to/your/qwen2.5VL model or verifier model"):
        self.device = device
        self.qwen_model_id = qwen_model_id

        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        if torch.cuda.is_available():
            torch.cuda.set_device(int(self.device.split(":")[-1]))

        self._load_models()

    def _load_models(self) -> None:
        t0 = time.time()
        print("[Service] Initializing mmgrounded-DINO ...")
        self.mmgdino_model = init_detector()

        print("[Service] Initializing GLM ...")
        self.GLM_pro, self.GLM_mod = initGLMT()

        print(f"[Service] Initializing Qwen VLM from: {self.qwen_model_id}")
        self.vlm_model, self.vlm_processor = load_qwen_vl_model(model_id=self.qwen_model_id, device=self.device)

        dt = time.time() - t0
        print(f"[Service] Models loaded. (t = {dt:.2f}s)  Using {self.device}")

    # ------------- Command Implementations -------------

    def cmd_plan(self, payload: Dict[str, Any], blobs: List[bytes]) -> Tuple[Dict[str, Any], List[bytes]]:
        """
        输入:
          payload: {"task": str}
          blobs: [ init_frame_image_bytes ]
        输出:
          {"ok": True, "plan": List[str], "objs": List[str], "locas": List[str]}, []
        """
        task: str = payload["task"]
        if not blobs:
            raise ValueError("plan: missing image blob")
        init_frame = _decode_image_from_bytes(blobs[0])

        plan, objs, locas = goforward(task, init_frame, self.GLM_pro, self.GLM_mod)

        return {"ok": True, "plan": plan, "objs": objs, "locas": locas}, []

    def cmd_detect(self, payload: Dict[str, Any], blobs: List[bytes]) -> Tuple[Dict[str, Any], List[bytes]]:
        """
        输入:
          payload: {"obj_prompts": List[str], "loc_prompts": List[str], "pred_thr": float (opt,0.3)}
          blobs: [ first_frame_image_bytes ]
        输出:
          {"ok": True}, [ pickle.dumps(dirt) ]
        """
        if not blobs:
            raise ValueError("detect: missing image blob")
        frame = _decode_image_from_bytes(blobs[0])

        obj_prompts = payload.get("obj_prompts", [])
        loc_prompts = payload.get("loc_prompts", [])
        pred_thr = float(payload.get("pred_thr", 0.3))
        web = payload.get("web", True)

        dirt, replace = mmgdino(
            GLM_mod=self.GLM_mod,
            GLM_pro=self.GLM_pro,
            frame=frame,
            obj_prompts=obj_prompts,
            loc_prompts=loc_prompts,
            inferencer=self.mmgdino_model,
            pred_thr=pred_thr,
            web=web,
        )

        dirt_blob = pickle.dumps(dirt, protocol=pickle.HIGHEST_PROTOCOL)
        replace_blob = pickle.dumps(replace, protocol=pickle.HIGHEST_PROTOCOL)
        return {"ok": True}, [dirt_blob, replace_blob]

    def cmd_verify(self, payload: Dict[str, Any], blobs: List[bytes]) -> Tuple[Dict[str, Any], List[bytes]]:
        """
        输入:
          payload: {"instruction": str}
          blobs: [ jpg_bytes_0_main, jpg_bytes_0_eih, jpg_bytes_1_main, jpg_bytes_1_eih, ... ]
                 （每对是 (main_img, eye_in_hand_img)）
        输出:
          {"ok": True, "complete": bool, "latency_ms": int}, []
        """
        instruction = payload["instruction"]
        if len(blobs) % 2 != 0 or len(blobs) == 0:
            raise ValueError("verify: blobs must be even and non-empty (pairs of images)")

        q = deque(maxlen=len(blobs) // 2)
        for i in range(0, len(blobs), 2):
            main_np = _decode_image_from_bytes(blobs[i])
            eih_np = _decode_image_from_bytes(blobs[i + 1])
            main_img = Image.fromarray(main_np)
            eih_img = Image.fromarray(eih_np)
            q.append((main_img, eih_img))

        is_subtask_complete = check_completion_with_qwen_vl(
            vlm_model=self.vlm_model,
            vlm_processor=self.vlm_processor,
            image_pair_queue=q,
            current_subtask_instruction=instruction
        )

        return {"ok": True, "complete": bool(is_subtask_complete)}, []

# ---------- Service Commands ----------
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=str, default="ipc:///tmp/vision_planner.sock",
                        help="ZeroMQ REP endpoint (ipc:// 或 tcp://)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--qwen_model_id", type=str,
                        default="/zhaohan/ZJX/Agentic_VLA/code/qwenft/LLaMA-Factory/result/merged")
    args = parser.parse_args()

    print(f"[Service] Starting VisionPlannerService on {args.endpoint}")
    service = VisionPlannerService(device=args.device, qwen_model_id=args.qwen_model_id)
    print("[Service] READY")

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    # 限制排队，避免积压
    sock.setsockopt(zmq.RCVHWM, 2)
    sock.setsockopt(zmq.SNDHWM, 2)
    sock.bind(args.endpoint)

    while True:
        try:
            parts = sock.recv_multipart()
            if not parts:
                sock.send_multipart([b'{"ok": false, "err":"empty"}'])
                continue

            header = json.loads(parts[0].decode("utf-8"))
            blobs = parts[1:]
            cmd = header.get("cmd")

            if cmd == "exit":
                body, bin_out = {"ok": True, "msg": "服务已退出"}, []
                resp = [json.dumps(body).encode("utf-8")]
                sock.send_multipart(resp)
                print("[Service] 收到退出命令，正在关闭...")
                sys.exit(0)

            t0 = time.time()
            if cmd == "ping":
                body, bin_out = {"ok": True, "pong": True}, []
            elif cmd == "plan":
                body, bin_out = service.cmd_plan(header, blobs)
            elif cmd == "detect":
                body, bin_out = service.cmd_detect(header, blobs)
            elif cmd == "verify":
                body, bin_out = service.cmd_verify(header, blobs)
            else:
                body, bin_out = {"ok": False, "err": f"unknown cmd: {cmd}"}, []


            body["latency_ms"] = body.get("latency_ms", int((time.time() - t0) * 1000))
            resp = [json.dumps(body).encode("utf-8")]
            for b in bin_out:
                resp.append(b)
            sock.send_multipart(resp)

        except Exception as e:
            tb = traceback.format_exc()
            err = {"ok": False, "err": str(e), "traceback": tb}
            sock.send_multipart([json.dumps(err).encode("utf-8")])


if __name__ == "__main__":
    main()