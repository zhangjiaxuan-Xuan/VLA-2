import os,re,time
import cv2
import json
from typing import Any, List, Optional, Union, Dict, Iterable, Tuple 
from mmdet.apis import DetInferencer
from mmdet.utils import register_all_modules
register_all_modules()
import torch
import nltk
from PIL import Image
import numpy as np
from auto_DL import search_reflect
# If NLTK support is required, set the data path (replace with your path)
nltk.data.path=["path/to/your/GroundingDINO/mmdetection/nltk_data"]
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ———— Configuration Section ————
CONFIG_FILE     = "path/to/your/GroundingDINO/mmdetection/usertest/results/libero_arg_testaim/grounding_dino_swin-t_finetune_8xb4_20e_cat_copy.py"
CHECKPOINT_FILE = "path/to/your/GroundingDINO/mmdetection/usertest/results/libero_arg_testaim/best_coco_bbox_mAP_epoch_99.pth"
color_json = "path/to/your/script/color.json" # or use default script/color.json
DEVICE          = "cuda:0" if torch.cuda.is_available() else "cpu"
# ——————————————————


Subtasks = Union[str, List[str]]
def inject_colors_into_tasklist(
    tasklist: Subtasks,
    dirt: List[Dict[str, Any]],
    *,
    # dirt schema keys (fixed to your structure)
    prompt_key: str = "prompt",
    color_name_key: str = "color_name",
    box_key: str = "box",
    # behavior
    skip_box_all_minus1: bool = True,           # skip entries with box == [-1,-1,-1,-1]
    case_sensitive: bool = False,               # case-insensitive by default
    word_boundary: bool = False,                # set True if you want whole-word matches only
    longest_first: bool = True,                 # replace longer prompts first
    dedupe_nested: bool = True,                 # drop shorter prompts contained by longer ones
    prefix_template: str = "{color} mask ",     # how to inject the prefix
    # idempotency: if a sentence already starts with "<color> mask ", skip injecting again
    existing_prefix_regex: Optional[str] = r"^\s*[A-Za-z]+\s+mask\s+",
) -> Tuple[Subtasks, Dict[str, Any]]:
    """
    Insert color_name prefixes into tasklist strings using `dirt`.
    Returns:
      new_tasklist (same type as input), report dict
    """

    # 1) Filter/collect prompt -> color_name from dirt
    def _valid_entry(d: Dict[str, Any]) -> bool:
        if prompt_key not in d or color_name_key not in d:
            return False
        if skip_box_all_minus1 and box_key in d and d.get(box_key) == [-1, -1, -1, -1]:
            return False
        p = str(d.get(prompt_key, "")).strip()
        c = str(d.get(color_name_key, "")).strip()
        return len(p) > 0 and len(c) > 0

    entries = [d for d in dirt if _valid_entry(d)]
    prompt2color: Dict[str, str] = {}
    for d in entries:
        # latest wins if prompts repeat
        prompt2color[str(d[prompt_key])] = str(d[color_name_key])

    if not prompt2color:
        return tasklist, {"applied": [], "skipped": ["no_valid_dirt"], "num_replacements": 0}

    # 2) Build prompt list and sort
    prompts = list(prompt2color.keys())
    if longest_first:
        prompts.sort(key=len, reverse=True)

    flags = 0 if case_sensitive else re.IGNORECASE  # case-insensitive matching is common.

    def _pattern_for(p: str) -> re.Pattern:
        # optional whole-word matching via \b boundaries.
        if word_boundary:
            return re.compile(rf"\b{re.escape(p)}\b", flags)
        else:
            return re.compile(re.escape(p), flags)

    def _already_prefixed(s: str) -> bool:
        return bool(existing_prefix_regex and re.search(existing_prefix_regex, s))

    # 3) Replace within one string
    def _replace_in_string(s: str) -> Tuple[str, List[Tuple[str, str, int]]]:
        ops: List[Tuple[str, str, int]] = []
        present = [p for p in prompts if _pattern_for(p).search(s)]

        if dedupe_nested and present:
            filtered = []
            for i, p in enumerate(present):
                keep = True
                for j, q in enumerate(present):
                    if i != j and len(q) > len(p) and q.find(p) != -1:
                        keep = False
                        break
                if keep:
                    filtered.append(p)
            present = filtered

        if _already_prefixed(s):
            return s, ops

        # Use re.subn to replace at most once per prompt and collect counts.
        for p in present:
            color = prompt2color[p]
            pat = _pattern_for(p)
            replacement = prefix_template.format(color=color) + p
            s, cnt = pat.subn(replacement, s, count=1)
            if cnt > 0:
                ops.append((p, color, cnt))
        return s, ops

    # 4) Dispatch by tasklist type
    applied_ops: List[Tuple[str, str, int]] = []
    skipped_reasons: List[str] = []

    if isinstance(tasklist, str):
        new_s, ops = _replace_in_string(tasklist)
        applied_ops.extend(ops)
        result: Subtasks = new_s

    elif isinstance(tasklist, list):
        new_list: List[str] = []
        for i, elem in enumerate(tasklist):
            if isinstance(elem, str):
                new_s, ops = _replace_in_string(elem)
                applied_ops.extend(ops)
                new_list.append(new_s)
            else:
                skipped_reasons.append(f"tasklist[{i}] not str")
                new_list.append(elem)
        result = new_list
    else:
        skipped_reasons.append(f"unsupported tasklist type: {type(tasklist)}")
        result = tasklist

    report = {
        "applied": applied_ops,
        "skipped": skipped_reasons,
        "num_replacements": sum(cnt for _, _, cnt in applied_ops),
    }
    return result, report

def load_color_json(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Read color.json and return a dictionary containing "objs" and "loca" lists.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "objs" not in data or "loca" not in data:
        raise ValueError("color.json must contain 'objs' and 'loca'")
    return data

def process_frame_prompts(
        GLM_mod,
        GLM_pro,
        frame: Any,
        obj_prompts: List[str],
        loc_prompts: List[str],
        inferencer: DetInferencer,
        color_config: Dict[str, List[Dict[str, Any]]],
        pred_score_thr: float = 0.15,
        web: bool = True,
        iou_thr: float = 0.5,
        max_per_prompt: int = 50,
        overlap_thr: float = 0.5
        ) -> tuple[List[Dict[str, Any]], dict]:
    """
    Perform detection and post-processing on a single image with a set of prompts (objs and loca), return dirt list.

    Args:
      frame:            OpenCV BGR ndarray or PIL.Image, supported by detect_boxes
      obj_prompts:      Object prompt list
      loc_prompts:      Location prompt list
      inferencer:       Initialized DetInferencer
      color_config:     Return value from load_color_json()
      pred_score_thr:   Score threshold for detect_boxes
      iou_thr:          NMS IoU threshold for detect_boxes
      max_per_prompt:   Number of boxes to keep per prompt for detect_boxes
      overlap_thr:      Cross-prompt overlap filtering threshold

    Returns:
      List of {
        "prompt": str,
        "box":    List[float] or [-1,-1,-1,-1],
        "score":  float,
        "color":  rgb str，
        "color_name": str color_name
      }
    """
    # 1) prepare prompts and color lists
    prompts = obj_prompts + loc_prompts
    obj_colors = color_config["objs"]
    loca_colors = color_config["loca"]

    # 2) Call detect_boxes (internal deduplication/NMS has been done)
    raw, replace = detect_boxes(
        GLM_mod,
        GLM_pro,
        inferencer,
        frame,
        prompts,
        pred_score_thr=pred_score_thr,
        iou_thr=iou_thr,
        max_per_prompt=max_per_prompt,
        web=web,
    )

    # 3) Collect all candidates and perform overlapping filtering across prompts in descending order of score
    candidates = [
        {"prompt": p, "box": d["box"], "score": d["score"]}
        for p, dets in raw.items() for d in dets
    ]
    kept: List[Dict[str, Any]] = []
    for cand in sorted(candidates, key=lambda x: x["score"], reverse=True):
        bi, ai = cand["box"], None
        ai = (bi[2] - bi[0]) * (bi[3] - bi[1])
        skip = False
        for k in kept:
            if k["prompt"] == cand["prompt"]:
                continue
            bj = k["box"]
            aj = (bj[2] - bj[0]) * (bj[3] - bj[1])
            xi0, yi0 = max(bi[0], bj[0]), max(bi[1], bj[1])
            xi1, yi1 = min(bi[2], bj[2]), min(bi[3], bj[3])
            if xi1 > xi0 and yi1 > yi0:
                inter = (xi1 - xi0) * (yi1 - yi0)
                if inter > overlap_thr * min(ai, aj):
                    skip = True
                    break
        if not skip:
            kept.append(cand)

    # 4) Grouping: same prompt may have multiple boxes
    grouped: Dict[str, List[Dict[str, Any]]] = {p: [] for p in prompts}
    for c in kept:
        grouped[c["prompt"]].append(c)

    # 5) Construct dirt: first objs, then loca
    dirt: List[Dict[str, Any]] = []
    for idx, p in enumerate(obj_prompts):
        rgb = obj_colors[idx]["rgb"] if idx < len(obj_colors) else obj_colors[-1]["rgb"]
        rgb_name = obj_colors[idx]["note"] if idx < len(obj_colors) else obj_colors[-1]["note"]
        boxes = grouped.get(p, [])
        if boxes:
            for c in boxes:
                dirt.append({
                    "prompt": p, "box": c["box"], "score": c["score"], "color": rgb,"color_name": rgb_name
                })
        else:
            dirt.append({"prompt": p, "box": [-1,-1,-1,-1], "score": 0.0, "color": rgb, "color_name": rgb_name})

    for idx, p in enumerate(loc_prompts):
        rgb = loca_colors[idx]["rgb"] if idx < len(loca_colors) else loca_colors[-1]["rgb"]
        rgb_name = loca_colors[idx]["note"] if idx < len(loca_colors) else loca_colors[-1]["note"]
        boxes = grouped.get(p, [])
        if boxes:
            for c in boxes:
                dirt.append({
                    "prompt": p, "box": c["box"], "score": c["score"], "color": rgb, "color_name": rgb_name
                })
        else:
            dirt.append({"prompt": p, "box": [-1,-1,-1,-1], "score": 0.0, "color": rgb, "color_name": rgb_name})

    return dirt, replace

def process_info_jsonl(info_jsonl: str,
                      color_json: str,
                      inferencer: DetInferencer,
                      GLM_pro: Any,
                      GLM_mod: Any,
                      **kwargs
                      ) -> List[Dict[str, Any]]:
    """
    Read the info.jsonl file, call process_frame_prompts in batch, and return the result for each record.

    Args:
        info_jsonl:   Path to the info.jsonl file
        color_json:   Path to the color JSON file
        inferencer:   An initialized DetInferencer
        **kwargs:     Additional keyword arguments passed to process_frame_prompts

    Returns:
        List of {"video": str, "dirt": [...]}
    """
    color_cfg = load_color_json(color_json)
    folder = os.path.dirname(info_jsonl)
    output = []

    with open(info_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            video = rec["video"]
            obj_prompts = rec.get("objs", [])
            loc_prompts = rec.get("location", [])
            # read video first frame
            cap = cv2.VideoCapture(os.path.join(folder, video))
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise RuntimeError(f"can not read video frame: {video}")

            dirt,replace = process_frame_prompts(
                frame=frame,
                obj_prompts=obj_prompts,
                loc_prompts=loc_prompts,
                inferencer=inferencer,
                color_config=color_cfg,
                GLM_pro=GLM_pro,
                GLM_mod=GLM_mod,
                **kwargs
            )
            output.append({"video": video, "dirt": dirt})

    return output

def mmgdino(
        GLM_mod,
        GLM_pro,
        frame: np.ndarray,
        obj_prompts: list,
        loc_prompts: list,
        inferencer: DetInferencer,
        pred_thr: float = 0.2,
        web: bool = True,
        **kwargs,
    ):
    color_config = load_color_json(color_json)
    dirt, replace = process_frame_prompts(
        GLM_mod,
        GLM_pro,
        frame=frame,
        obj_prompts=obj_prompts,
        loc_prompts=loc_prompts,
        inferencer=inferencer,
        color_config=color_config,
        pred_score_thr = pred_thr,
        web=web,
        **kwargs
    )
    return dirt, replace

def init_detector(config: str = CONFIG_FILE,
                  checkpoint: str = CHECKPOINT_FILE,
                  device: str = DEVICE) -> DetInferencer:
    """
    init and return DetInferencer.
    """
    return DetInferencer(
        config,
        checkpoint,
        device=device
    )

def _prep_image(img):
    """
    Internal helper: normalize three input types into an OpenCV BGR ndarray,
    and return two values (infer_input, draw_img).
    infer_input is passed to DetInferencer and can be a str(path) or an RGB ndarray,
    draw_img is always a BGR ndarray used for visualization.
    """
    # 1) If it's a file path
    if isinstance(img, str):
        bgr = cv2.imread(img)
        # inferencer supports reading the path directly
        return img, bgr

    # 2) If it's a PIL Image
    if isinstance(img, Image.Image):
        rgb = np.array(img)
        # inferencer generally expects an RGB ndarray
        infer_np = rgb.copy()
        # convert to BGR for cv2 drawing
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return infer_np, bgr

    # 3) If it's an OpenCV ndarray
    if isinstance(img, np.ndarray):
        # assume the input is a BGR ndarray (cv2's format)
        bgr = img.copy()
        # convert to RGB for the inferencer
        infer_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return infer_np, bgr

    raise TypeError(f"Unsupported image type: {type(img)}")

import unicodedata
import importlib

def replace_in_mem_test(
    *,
    boxes: Union[List[List[float]], "np.ndarray", "torch.Tensor", None],
    score: Union[List[float], "np.ndarray", "torch.Tensor", None],
    image: Union["np.ndarray", "Image.Image", None],
    comimage: Union[Optional["Image.Image"],str,List[str],None],
    prompt: str,
    keywords: Optional[List[str]],
    GLM_pro: Any,   # == processor_GLM
    GLM_mod: Any,   # == model_GLM
    web: bool = True
) -> Optional[str]:
    """
    Simplified 'memory-first' replace:
      1) If 'prompt' is already in /mnt/data/replace_mem.json, return saved replacement.
      2) Else, construct messages according to the 3 cases:
         (A) no comimage and no keywords
         (B) no comimage, no keywords, no boxes/score
         (C) all info present
         - If boxes/score present, crop the highest-score bbox from 'image' and attach as its own user message.
         - If comimage present, attach as its own user message.
         - If keywords present, add a user text message listing them.
         - If web=True, fetch brief external snippets and add as user text message.
      3) Ask GLM (via your 'useglmt') to output EXACTLY ONE label from knownlist.json or 'NONE'.
      4) If label != 'NONE', persist mapping {prompt: {"replace": label}} into replace_mem.json and return label.
    """
    


    # ------------------------- small helpers -------------------------
    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKC", s)
        s = "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")
        return re.sub(r"\s+", " ", s.strip())

    def _load_json(path: str, default):
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return default

    def _save_json(path: str, data):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # optional deps
    np = None
    torch = None
    try:
        np = importlib.import_module("numpy")
    except Exception:
        pass
    try:
        torch = importlib.import_module("torch")
    except Exception:
        pass
    PIL_Image = None
    try:
        from PIL import Image
        PIL_Image = Image
    except Exception:
        pass

    if comimage is not None and isinstance(comimage, (str,list)) and len(comimage) > 0:
        comimage = Image.open(comimage) if isinstance(comimage, str) else Image.open(comimage[0])

    # ------------------------- memory first -------------------------
    mem_path = "path/to/your/simple_exp/replace_mem.json"
    known_path = "path/to/your/agentic-robot/script/knownlist.json"

    mem = _load_json(mem_path, {})
    known = _load_json(known_path, {})
    known_list: List[str] = list(known.get("known", []))

    if not isinstance(prompt, str) or not prompt.strip():
        return None
    if not known_list:
        return None

    norm_prompt = _norm(prompt)
    if norm_prompt in mem and isinstance(mem[norm_prompt], dict) and "replace" in mem[norm_prompt]:
        # found cached mapping
        lab = str(mem[norm_prompt]["replace"])
        return lab if lab in known_list else None

    # ------------------------- to PIL helpers -------------------------
    def to_pil(img):
        if img is None or PIL_Image is None:
            return None
        if isinstance(img, PIL_Image.Image):
            return img
        if np is not None and isinstance(img, np.ndarray):
            try:
                a = img
                if a.dtype != np.uint8:
                    # heuristic: normalize floats in [0,1] to uint8
                    m = float(getattr(a, "max", lambda: 255)())
                    if m <= 1.0:
                        a = (a * 255.0).clip(0, 255)
                    a = a.astype("uint8")
                if a.ndim == 2:
                    return PIL_Image.fromarray(a, mode="L")
                if a.ndim == 3:
                    if a.shape[2] == 1:
                        return PIL_Image.fromarray(a[..., 0], mode="L")
                    return PIL_Image.fromarray(a)
            except Exception:
                return None
        if torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(img):
            try:
                return to_pil(img.detach().cpu().numpy())
            except Exception:
                return None
        return None

    pil_image = to_pil(image)
    pil_com = to_pil(comimage)

    # ------------------------- boxes / crop -------------------------
    def to_list(x):
        if x is None:
            return []
        if np is not None and isinstance(x, np.ndarray):
            try:
                return x.tolist()
            except Exception:
                return []
        if torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(x):
            try:
                return x.detach().cpu().tolist()
            except Exception:
                return []
        return x
    
    
    def _round_up(x: int, factor: int) -> int:
        return int(((x + factor - 1) // factor) * factor)

    def ensure_min_size(img: Image.Image, *, min_side: int = 28, multiple: int = 28,
                        max_side: int = 1024, fill=(0, 0, 0)) -> Image.Image:
        """
        Scale up / center-pad the image so that the shortest side >= min_side,
        and try to align dimensions to a multiple of `multiple`.
        This avoids preprocessing errors like "height/width must be larger than factor: 28".
        """
        w, h = img.size
        # Target size: ensure shortest side >= min_side, then round up to a multiple
        tw = min(_round_up(max(w, min_side), multiple), max_side)
        th = min(_round_up(max(h, min_side), multiple), max_side)
        if (w >= min_side and h >= min_side) and (w == tw and h == th):
            return img  # already satisfies requirements

        # Scale proportionally to fit within the target canvas, then paste centered onto the canvas
        scale = min(tw / w, th / h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = img.resize((nw, nh), Image.BICUBIC)
        canvas = Image.new("RGB", (tw, th), fill)
        canvas.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
        return canvas

    boxes_list = to_list(boxes) or []
    scores_list = to_list(score) or []
    if not isinstance(boxes_list, list) or not all(isinstance(b, (list, tuple)) and len(b) >= 4 for b in boxes_list):
        boxes_list = []
    if not scores_list or len(scores_list) != len(boxes_list):
        scores_list = [1.0] * len(boxes_list)

    def norm_boxes_to_pixels(b: List[float], W: int, H: int) -> Tuple[int,int,int,int]:
        try:
            x, y, c, d = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            if c > 1 or d > 1:
                # pixel: treat as [x1,y1,x2,y2] or [x,y,w,h]
                if c > x and d > y: x1,y1,x2,y2 = x,y,c,d
                else: x1,y1,x2,y2 = x,y,(x+c),(y+d)
            else:
                # normalized
                if c > x and d > y: x1,y1,x2,y2 = x*W,y*H,c*W,d*H
                else: x1,y1,x2,y2 = x*W,y*H,(x+c)*W,(y+d)*H
        except Exception:
            return (0,0,W,H)
        x1,y1,x2,y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        x1 = max(0, min(W-1, x1)); y1 = max(0, min(H-1, y1))
        x2 = max(0, min(W,   x2)); y2 = max(0, min(H,   y2))
        if x2 <= x1 or y2 <= y1:
            return (0,0,W,H)
        return (x1,y1,x2,y2)

    top_crop = None
    if pil_image is not None and boxes_list:
        W, H = pil_image.size
        # pick highest-score bbox
        best_idx = max(range(len(boxes_list)), key=lambda i: float(scores_list[i]))
        x1, y1, x2, y2 = norm_boxes_to_pixels(boxes_list[best_idx], W, H)
        try:
            cimg = pil_image.crop((x1, y1, x2, y2))
            if cimg and cimg.width >= 8 and cimg.height >= 8:
                # added resize safeguard at least 28x28, or bug may occur in some cases
                if cimg.width < 28 or cimg.height < 28:
                    cimg = ensure_min_size(cimg, min_side=28, multiple=28)
                top_crop = cimg
        except Exception:
            top_crop = None


    # ------------------------- search snippets (optional) -------------------------
    def fetch_snippets(queries: List[str], limit: int = 4) -> str:
        """
        Try 'wikipedia' then 'wikipediaapi'; return a short bullet text.
        """
        out = []
        # wikipedia
        try:
            wikipedia = importlib.import_module("wikipedia")
            try:
                wikipedia.set_lang("en")
            except Exception:
                pass
            for q in queries:
                try:
                    titles = wikipedia.search(q, results=1) or []
                    for t in titles:
                        try:
                            page = wikipedia.page(t, auto_suggest=False, preload=False)
                            s = (page.summary or "").strip().replace("\n", " ")
                            if s:
                                out.append(f"- {t}: {s[:280]}")
                        except Exception:
                            continue
                except Exception:
                    continue
        except Exception:
            pass
        # wikipediaapi
        if len(out) < limit:
            try:
                wikipediaapi = importlib.import_module("wikipediaapi").Wikipedia
                for lang in ("en","zh"):
                    try:
                        wiki = wikipediaapi(lang)
                    except Exception:
                        continue
                    for q in queries:
                        try:
                            page = wiki.page(q)
                            if page.exists():
                                s = (page.summary or "").strip().replace("\n", " ")
                                if s:
                                    out.append(f"- {page.title} ({lang}): {s[:280]}")
                        except Exception:
                            continue
            except Exception:
                pass
        # trim
        return "\n".join(out[:limit])

    # ------------------------- build messages (1 image per user message) -------------------------
    # Rules:
    #   - Print the allowed vocabulary, ask to pick EXACTLY ONE or 'NONE'
    #   - Each image goes in a separate user message
    #   - Separate user messages for: prompt, keywords, web snippets, comimage, crop
    allowed_text = "\n".join(f"- {lab}" for lab in known_list)

    messages = []
    # system steer
    system_msg = {'role': 'system', 'content': [
    {'type': 'text', 'text': 'You normalize open-world object mentions to a closed training vocabulary. '
                             'Return EXACTLY ONE label copied verbatim from the allowed list below, '
                             'or rarely output NONE if no label applies.'
                             }
    ]}

    messages.append(system_msg)

    # user: allowed vocab
    messages.append({
        "role": "user",
        "content": [
            {"type": "text",
             "text": "Allowed vocabulary:\n" + allowed_text}
        ]
    })

    # Determine case
    has_com = pil_com is not None
    has_kw = bool(keywords)
    has_boxes = bool(top_crop is not None)
    has_scores = bool(boxes_list)

    # shared: prompt text
    messages.append({
        "role": "user",
        "content": [{"type":"text","text": f"New object mention: {norm_prompt}"}]
    })

    # Case-specific assembly
    # (A) no comimage and no keywords
    if (not has_com) and (not has_kw) and (has_boxes or (pil_image is not None)):
        if has_boxes and top_crop is not None:
            messages.append({
                "role": "user",
                "content": [
                    {"type":"text","text":"Evidence crop (highest detector score)."},
                    {"type":"image","image": top_crop},
                ],
            })
        elif pil_image is not None:
            messages.append({
                "role": "user",
                "content": [
                    {"type":"text","text":"Context image."},
                    {"type":"image","image": pil_image},
                ],
            })

    # (B) no comimage, no keywords, no boxes/score
    if (not has_com) and (not has_kw) and (not has_boxes) and (not has_scores):
        # only text + maybe the raw image if given
        if pil_image is not None:
            messages.append({
                "role": "user",
                "content": [
                    {"type":"text","text":"Context image."},
                    {"type":"image","image": pil_image},
                ],
            })

    # (C) all information present (interpretation: comimage & keywords & boxes/score available)
    if has_com and has_kw and has_boxes:
        # comimage as its own user turn
        messages.append({
            "role": "user",
            "content": [
                {"type":"text","text":"Composite reference image from the web."},
                {"type":"image","image": pil_com},
            ],
        })
        # crop as its own user turn
        messages.append({
            "role": "user",
            "content": [
                {"type":"text","text":"Top-scoring evidence crop from the original image."},
                {"type":"image","image": top_crop},
            ],
        })
        # keywords as text
        messages.append({
            "role": "user",
            "content": [{"type":"text","text": "Image/scene keywords: " + ", ".join(map(str, keywords))}]
        })

    # web snippets (optional)
    if web:
        qs = [norm_prompt]
        if keywords:
            qs.extend([_norm(k) for k in keywords if isinstance(k, str)])
        web_brief = fetch_snippets(qs, limit=7)
        if web_brief:
            messages.append({
                "role": "user",
                "content": [{"type":"text","text": "External brief (web/Wikipedia):\n" + web_brief}]
            })

    # final instruction: pick one
    messages.append({
        "role": "user",
        "content": [{"type":"text","text": "Return exactly one label from the allowed list above, or `NONE`."}]
    })

    # ------------------------- call your GLM wrapper -------------------------
    # useglmt(messages, processor_GLM, model_GLM) must be defined in your codebase

    try:
        output = useglmt(messages, GLM_pro, GLM_mod)  # GLM_pro=processor, GLM_mod=model
    except Exception as e:
        print(f"replace_in_mem_test: useglmt() failed: {e}")
        return None

    # ------------------------- parse model output to allowed label -------------------------
    def pick_label(text: Optional[str], allowed: List[str]) -> Optional[str]:
        if not text:
            return None

        # 1) Extract candidate "box" text block (prefer begin_of_box inside <answer> if present)
        t = str(text)

        # match <answer>...</answer>
        ans = re.search(r"<answer\b[^>]*>([\s\S]*?)</answer\s*>",
                        t, flags=re.IGNORECASE | re.DOTALL)

        def _first_box_block(s: str) -> Optional[str]:
            """Find the first <begin_of_box> ... <end_of_box> content in s"""
            m = re.search(r"<\|\s*begin_of_box\s*\|>([\s\S]*?)<\|\s*end_of_box\s*\|>",
                          s, flags=re.IGNORECASE | re.DOTALL)
            return m.group(1) if m else None

        # Priority: box nested inside <answer> > plain content of <answer> > first box in the whole text
        candidate_block = None
        if ans:
            inner = ans.group(1)
            candidate_block = _first_box_block(inner) or inner
        else:
            candidate_block = _first_box_block(t)

        if not candidate_block:
            # No box block found: per the new rule, give up (do not fallback to old substring/quote-tolerant logic)
            return None
        return candidate_block.strip()

    try:
        label = pick_label(output, known_list)
    except Exception as e:
        print(f"replace_in_mem_test: pick_label() failed: {e}")
        return None
    if label is None:
        return None

    # ------------------------- persist and return -------------------------
    if label != "NONE":
        # store mapping for future fast return
        mem[_norm(prompt)] = {"replace": label}
        try:
            _save_json(mem_path, mem)
        except Exception:
            pass
        return label
    else:
        return None



import torchvision.ops as tv_ops
def detect_boxes(
        GLM_mod,
        GLM_pro,
        inferencer: DetInferencer,
        img_path: str,
        prompts: list,
        pred_score_thr: float = 0.15,
        iou_thr: float = 0.5,
        web: bool = True,
        max_per_prompt: int = 50) -> tuple[dict, dict]:
    """
    Run inference on a single image with multiple prompts, returning detected boxes and
    scores for each prompt.

    Args:
      inferencer: an initialized DetInferencer instance
      img_path:   path to the image
      prompts:    list of strings, each representing a detection prompt
      pred_score_thr: confidence score threshold

    Returns:
      dict mapping each prompt to a list of dicts, each dict contains:
        - 'box': [x0, y0, x1, y1]
        - 'score': float

    Note: The bounding boxes returned by this function are filtered per prompt.
    Cross-prompt filtering (e.g., removing overlapping detections across different prompts)
    is not performed here and should be handled in a subsequent processing step.
    """
    results = {}
    replace_map = {}
    device = torch.device(DEVICE)
    image,bgr = _prep_image(img_path) 
    for prompt in prompts:
        # if not prompt or not isinstance(prompt, str):continue
        # print(f"🔍 processing prompt: {prompt}")
        # if prompt =="MILK  box":
        #     a=1
        # use inferencer，no image and save
        # prompt = prompt.lower()
        replace_map[prompt] = {"org": prompt, "replace": None}
        try:
            outs = inferencer(
                image,
                texts=[prompt],
                draw_pred=False,
                pred_score_thr=pred_score_thr,
            )
        except Exception as e:
            continue
        # inferencer returns mmdet's Output object, where .pred_instances contains boxes and scores

        pred = outs["predictions"][0]
        raw_boxes  = pred["bboxes"]   # list of [x0,y0,x1,y1]
        raw_scores = pred["scores"]   # list of scores

        # If nothing is detected, return empty directly

            # if have :go else: results[prompt] = [] contine

        # 2) Convert to Tensor and initially filter by score
        boxes  = torch.tensor(raw_boxes,  dtype=torch.float32, device=device)
        scores = torch.tensor(raw_scores, dtype=torch.float32, device=device)
        keep_mask = scores > pred_score_thr
        boxes, scores = boxes[keep_mask], scores[keep_mask]
        if boxes.numel() != 0:
            language_start = time.perf_counter()
            replace_word = replace_in_mem_test(boxes=boxes,score=scores,image=image, comimage=None, prompt=prompt, keywords=None, GLM_pro=GLM_pro, GLM_mod=GLM_mod,web=web)
            language_end = time.perf_counter()
            if isinstance(replace_word, str) and replace_word.strip():
                replace_map[prompt]["replace"] = replace_word.strip()
        if len(boxes) == 0 and web:
            #search for this prompt
            print(f"Arg for the prompt:{prompt}")
            keywords,comimage = search_reflect(image,prompt,GLM_pro,GLM_mod) ##PIL format composite image
            # with more words
            new_prompt = prompt + ',' + ' '.join(keywords) + '.'
            # inference again
            try:
                outs = inferencer(
                    image,
                    texts=[new_prompt],
                    draw_pred=False,
                    pred_score_thr=pred_score_thr,
                )
            except Exception as e:
                continue
            pred = outs["predictions"][0]
            raw_boxes  = pred["bboxes"]   # list of [x0,y0,x1,y1]
            raw_scores = pred["scores"]   # list of scores

            # If nothing is detected, return empty directly

                # if have :go else: results[prompt] = [] contine

            # 2) Convert to Tensor and initially filter by score
            boxes  = torch.tensor(raw_boxes,  dtype=torch.float32, device=device)
            scores = torch.tensor(raw_scores, dtype=torch.float32, device=device)
            keep_mask = scores > round(pred_score_thr / 1.5, 3)
            boxes, scores = boxes[keep_mask], scores[keep_mask]

            if boxes.numel() != 0:
                language_start = time.perf_counter()
                replace_word = replace_in_mem_test(boxes=boxes,score=scores,image=image, comimage=comimage, prompt=prompt, keywords=keywords, GLM_pro=GLM_pro, GLM_mod=GLM_mod,web=web)
                language_end = time.perf_counter()
                if isinstance(replace_word, str) and replace_word.strip():
                    replace_map[prompt]["replace"] = replace_word.strip()
        if boxes.numel() == 0:
            language_start = time.perf_counter()
            replace_word = replace_in_mem_test(boxes=[],score=[],image=image, comimage=None, prompt=prompt, keywords=None, GLM_pro=GLM_pro, GLM_mod=GLM_mod,web=web)
            language_end = time.perf_counter()
            if isinstance(replace_word, str) and replace_word.strip():
                replace_map[prompt]["replace"] = replace_word.strip()
            results[prompt] = []
            continue
        
        language_time = language_end - language_start
        print(f"Language model time: {language_time:.3f} seconds, prompt: {prompt}, replace: {replace_word}")
        # — Dynamic adjustment of max_per_prompt —
        # num_candidates = scores.shape[0]
        # dynamic_k = min(max_per_prompt, num_candidates)
        # if dynamic_k <= 0:
        #     results[prompt] = []
        #     continue
        
        # 3) NMS on GPU
        keep_idx = tv_ops.nms(boxes, scores, iou_thr)

        # 4) Top-K truncation
        keep_idx = keep_idx[:max_per_prompt]

        # 5) Convert back to Python list
        dets = []
        for idx in keep_idx:
            b = boxes[idx].tolist()
            s = scores[idx].item()
            dets.append({'box': b, 'score': s})
        areas = [
            (b[2] - b[0]) * (b[3] - b[1])
            for b in (d['box'] for d in dets)
        ]
        # 2) Get indices in descending order by area
        order = sorted(range(len(dets)), key=lambda i: areas[i], reverse=True)
        filtered = []
        removed = set()
        # 3) Iterate: when encountering large boxes, remove all smaller boxes that overlap more than 0.25 of the small box's area
        for rank, i in enumerate(order):
            if i in removed:
                continue
            filtered.append(dets[i])
            x0_i, y0_i, x1_i, y1_i = dets[i]['box']
            for j in order[rank+1:]:
                if j in removed:
                    continue
                x0_j, y0_j, x1_j, y1_j = dets[j]['box']
                # Calculate intersection area
                xi0 = max(x0_i, x0_j)
                yi0 = max(y0_i, y0_j)
                xi1 = min(x1_i, x1_j)
                yi1 = min(y1_i, y1_j)
                if xi1 > xi0 and yi1 > yi0:
                    inter = (xi1 - xi0) * (yi1 - yi0)
                    # If intersection area exceeds 0.25 of small box area, discard small box
                    if inter > 0.25 * areas[j]:
                        removed.add(j)

        # Round all box floating point numbers to integers
        for det in filtered:
            det['box'] = [int(round(x)) for x in det['box']]
        results[prompt] = filtered
    ## if none box for a prompt or score is accept but low, search for more info and bbox again.

    return results, replace_map


def draw_detections(img_path: str,
                    detections: dict,
                    save_path: str,
                    box_color=(0,255,0),
                    thickness=2,
                    font_scale=0.2):  # Smaller font size
    """
    Draw the results of detect_boxes on the image and save.

    Args:
      img_path:    Original image path
      detections:  Dict returned by detect_boxes
      save_path:   Path to save the image with bounding boxes
    """
    rgb, bgr = _prep_image(img_path)  # Get BGR image for drawing
    for prompt, dets in detections.items():
        for det in dets:
            x0, y0, x1, y1 = map(int, det['box'])
            cv2.rectangle(bgr, (x0, y0), (x1, y1), box_color, thickness)
            label = f"{prompt}:{det['score']:.2f}"
            cv2.putText(
                bgr, label,
                (x0, y0-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, box_color, thickness
            )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, bgr)

def useglmt(
        messages,
        processor_GLM,
        model_GLM,
    ):
    """
    Use GLM-4.1V-9B-Thinking model for image description.
    """

    inputs = processor_GLM.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model_GLM.device)
    generated_ids = model_GLM.generate(**inputs, max_new_tokens=2000, do_sample=False, repetition_penalty=1.25).to("cuda:0")
    output_text = processor_GLM.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    return output_text

# ———— Usage Examples ————
if __name__ == "__main__":
    detector = init_detector()
    # image_file = "path/to/your/testcode/testimage/labubu.jpg"
    # prompts = ["ball, left"]
    # # 1) Get box list for each prompt
    # dets = detect_boxes(detector, image_file, prompts, pred_score_thr=0.2)
    # print(dets)  # {'green tennis ball': [...], 'red cup': [...]}

    # # 2) If you want to visualize, call draw_detections
    # out_image = "path/to/your/GroundingDINO/result/vis.jpg"
    # draw_detections(image_file, dets, out_image)
    # print(f"Saved visualization to {out_image}")

    test_jsonl = "path/to/your/LIBERO/datasets-copy/00_all/libero_10_no_noops/info_test.jsonl"
    color_json = "path/to/your/src/main/color.json"
    out_vis_dir = "path/to/your/script/test/testimage"
    # Process JSONL, get dirt for each video
    results = process_info_jsonl(
        info_jsonl=test_jsonl,
        color_json=color_json,
        inferencer=detector,
        pred_score_thr=0.2,
        iou_thr=0.5,
        max_per_prompt=50,
        overlap_thr=0.5
    )

    folder = os.path.dirname(test_jsonl)
    for item in results:
        video = item["video"]
        dirt  = item["dirt"]

        # 1) Read first frame
        video_path = os.path.join(folder, video)
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"Failed to read frame for {video}")
            continue

        # 2) Construct detections dict required by draw_detections
        #    prompt -> list of {'box':..., 'score':...}
        detections = {}
        for entry in dirt:
            p = entry["prompt"]
            b = entry["box"]
            s = entry["score"]
            # Ensure list exists
            detections.setdefault(p, [])
            # Even if it's an empty placeholder box, pass it in for annotation
            detections[p].append({"box": b, "score": s})

        # 3) Save visualization results
        save_path = os.path.join(out_vis_dir, video.replace(".mp4", ".jpg"))
        draw_detections(
            frame,           # draw_detections supports ndarray or path
            detections,
            save_path,
            box_color=None,  # If draw_detections supports per-entry color, this can be ignored
            thickness=2,
            font_scale=0.6
        )
        print(f"Saved visualization for {video} → {save_path}")
    