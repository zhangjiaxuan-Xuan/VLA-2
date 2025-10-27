# ==== Smoke Test for replace_in_mem (revised: always use info bbox + search_reflect) ====
# Assumptions:
# - You already defined: replace_in_mem(), GLM_pro (processor), GLM_mod (model)
# - You already have: search_reflect()  -> returns (keywords: List[str], comimage: PIL.Image or None)
# - Files present:
#     INFO_PATH         # test items: { "info1": {...}, "info2": {...}, ... }
#     KNOWN_PATH        # { "known": ["wine bottle", "black bowl", ...] }
#     MEM_PATH          # memory cache (updated by replace_in_mem)
# ============================================================================

import os, json, sys
from typing import Any, Dict, List, Optional

import re
import cv2
from typing import Any, List, Optional, Union, Dict, Iterable, Tuple
import torch
import nltk
from PIL import Image
import numpy as np
import time

# your libs
sys.path.append("path/to/your/script")
from auto_DL import search_reflect, initGLMT
from mmdet.utils import register_all_modules
register_all_modules()


from mmgdino import replace_in_mem_test

# Configuration section (adjust paths to your project as needed)
CONFIG_FILE     = "path/to/your/GroundingDINO/config.py"
CHECKPOINT_FILE = "path/to/your/GroundingDINO/checkpoint.pth"
color_json      = "path/to/your/src/main/color.json"
DEVICE          = "cuda:0" if torch.cuda.is_available() else "cpu"

INFO_PATH   = "path/to/your/script/test_images/info.json"
KNOWN_PATH  = "path/to/your/script/knownlist.json"
MEM_PATH    = "path/to/your/simple_exp/replace_mem.json"

# If NLTK support is required, configure a local nltk_data path
nltk.data.path=["path/to/your/GroundingDINO/nltk_data"]
os.environ["TRANSFORMERS_OFFLINE"] = "1"

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

# 读取测试集与词表
info  = _load_json(INFO_PATH, {})
known = _load_json(KNOWN_PATH, {})
known_list: List[str] = list(known.get("known", []))

if not info:
    print(f"[SmokeTest] {INFO_PATH} is empty or missing.")
    sys.exit(1)
if not known_list:
    print(f"[SmokeTest] {KNOWN_PATH} has no 'known' list.")
    sys.exit(1)

# Try loading the image if info.json provides an image_path
def _maybe_load_image(path: Optional[str]):
    if not path or not os.path.exists(path):
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None

pil_image = _maybe_load_image(info.get("image_path"))

web = True  # 允许在 replace_in_mem 内部追加外部摘要等（由你的实现决定）

results: Dict[str, Dict[str, Any]] = {}
hit = 0
total = 0

GLM_pro, GLM_mod = initGLMT()

# 遍历 infoX：统一流程 ——> 先 search_reflect 得到 (keywords, comimage)，
# 然后**始终**使用 info.json 提供的 bbox/score 调用 replace_in_mem。
for key in sorted(k for k in info.keys() if k.startswith("info")):
    
    case = info[key]
    prompt: str = case.get("name", "")
    aim: str    = case.get("aim", "NONE")
    boxes  = case.get("bbox", []) or []
    scores = case.get("score", []) or []

    # Load a per-sample image (if each infoX provides its own path)
    local_img_path = case.get("image_path")
    this_image = _maybe_load_image(local_img_path) if local_img_path else pil_image

    # Unified: run search_reflect for each sample
    keywords, comimage = [], None
    try:
        t_search0 = time.perf_counter()
        kw, com = search_reflect(this_image, prompt, GLM_pro, GLM_mod)  # -> (List[str], PIL.Image or None)
        t_search1 = time.perf_counter()
        search_time = t_search1 - t_search0
        if isinstance(kw, list):
            keywords = [str(x) for x in kw]
        if isinstance(com, Image.Image) or isinstance(com, str) or isinstance(com, list):
            comimage = com
    except Exception as e:
        search_time = None
        print(f"[{key}] search_reflect error: {e}")

    # Use bbox/score from info directly; if empty, pass empty lists so replace_in_mem can use fallback logic
    out_label = None
    try:
        t_rep0 = time.perf_counter()
        out_label = replace_in_mem_test(
            boxes=boxes,
            score=scores,
            image=this_image,
            comimage=comimage,
            prompt=prompt,
            keywords=keywords if keywords else None,
            GLM_pro=GLM_pro,
            GLM_mod=GLM_mod,
            web=web
        )
        t_rep1 = time.perf_counter()
        replace_time = t_rep1 - t_rep0
    except Exception as e:
        replace_time = None
        print(f"[{key}] replace_in_mem error: {e}")

    # ---- Record and evaluate ----
    total += 1
    ok = (out_label == aim) or (aim.upper() == "NONE" and (out_label is None))
    if ok: hit += 1
    results[key] = {
        "prompt": prompt,
        "aim": aim,
        "got": out_label,
        "ok": bool(ok),
        "keywords": keywords,
        "has_comimage": bool(comimage is not None),
        "num_boxes": len(boxes),
        "timings": {
            "search": search_time,
            "replace": replace_time
        }
    }
    print(f"[{key}] prompt='{prompt}' | aim='{aim}' | got='{out_label}' | "
          f"kw={len(keywords)} | com={'Y' if comimage else 'N'} | boxes={len(boxes)} | "
          f"search={results[key]['timings']['search']:.3f}s replace={results[key]['timings']['replace']:.3f}s | {'OK' if ok else 'MISS'}")

# Simple summary
print(f"[SmokeTest] Accuracy: {hit}/{total} = {hit/total:.2%}")

# (optional) write run results back to a file for inspection
_save_json("path/to/your/script/test_images/smoke_results.json", results)
# =============================================================================
