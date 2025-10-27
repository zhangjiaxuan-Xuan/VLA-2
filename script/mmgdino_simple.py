import os,re
import json
from typing import Any, Dict, List, Tuple, Union, Optional
import torch
import nltk
# If NLTK support is required, set the data path (replace with your path)
nltk.data.path=["path/to/your/GroundingDINO/mmdetection/nltk_data"]
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ———— Configuration Section ————
CONFIG_FILE     = "path/to/your/GroundingDINO/mmdetection/usertest/results/libero_arg_testaim/grounding_dino_swin-t_finetune_8xb4_20e_cat_copy.py"
CHECKPOINT_FILE = "path/to/your/GroundingDINO/mmdetection/usertest/results/libero_arg_testaim/best_coco_bbox_mAP_epoch_99.pth"
color_json = "path/to/your/src/main/color.json"
DEVICE          = "cuda:0" if torch.cuda.is_available() else "cpu"
# ——————————————————


Subtasks = Union[str, List[str]]
def inject_colors_into_tasklist(
    tasklist: Subtasks,
    dirt: List[Dict[str, Any]],
    replace_or_not: bool,
    replace: List[Dict[str, Any]],
    *,
    # dirt schema keys (fixed to your structure)
    prompt_key: str = "prompt",
    color_name_key: str = "color_name",
    box_key: str = "box",
    # behavior
    skip_box_all_minus1: bool = True,
    case_sensitive: bool = False,
    word_boundary: bool = False,
    longest_first: bool = True,
    dedupe_nested: bool = True,
    prefix_template: str = "{color} mask ",
    existing_prefix_regex: Optional[str] = r"^\s*[A-Za-z]+\s+mask\s+",
) -> Tuple[Subtasks, Dict[str, Any]]:
    """
    Insert color_name prefixes into tasklist strings using `dirt`,
    then (optionally) replace original prompts with refined ones from `replace`.
    Returns: new_tasklist, report dict
    """
    # ----------------------- (Original Logic: Color Injection) -----------------------
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
        prompt2color[str(d[prompt_key])] = str(d[color_name_key])

    if not prompt2color:
        return tasklist, {"applied": [], "skipped": ["no_valid_dirt"], "num_replacements": 0}

    prompts = list(prompt2color.keys())
    if longest_first:
        prompts.sort(key=len, reverse=True)

    flags = 0 if case_sensitive else re.IGNORECASE

    def _pattern_for(p: str) -> re.Pattern:
        if word_boundary:
            return re.compile(rf"\b{re.escape(p)}\b", flags)
        else:
            return re.compile(re.escape(p), flags)

    def _already_prefixed(s: str) -> bool:
        return bool(existing_prefix_regex and re.search(existing_prefix_regex, s))

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

        for p in present:
            color = prompt2color[p]
            pat = _pattern_for(p)
            replacement = prefix_template.format(color=color) + p
            s, cnt = pat.subn(replacement, s, count=1)
            if cnt > 0:
                ops.append((p, color, cnt))
        return s, ops

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

    # ----------------------- (New Logic: Replace prompt based on replace mapping) -----------------------
    if replace_or_not:
        # 1) Normalize replace as (org, repl) pairs; only keep items where repl is non-empty and different from org
        replace_pairs: List[Tuple[str, str]] = []
        try:
            if isinstance(replace, dict):
                # Compatible with {prompt: {"org": prompt, "replace": ...}}
                for k, v in replace.items():
                    org = (v.get("org") if isinstance(v, dict) else k) if v is not None else k
                    repl = v.get("replace") if isinstance(v, dict) else None
                    org = str(org).strip() if org is not None else ""
                    repl = str(repl).strip() if repl is not None else ""
                    if org and repl and repl != org:
                        replace_pairs.append((org, repl))
            else:
                # Compatible with List[Dict], keys may be org/replace or prompt/replace
                for it in (replace or []):
                    if not isinstance(it, dict): 
                        continue
                    org = str(it.get("org", it.get("prompt", ""))).strip()
                    repl = str(it.get("replace", "")).strip()
                    if org and repl and repl != org:
                        replace_pairs.append((org, repl))
        except Exception:
            # If parsing fails, skip the replacement phase directly
            replace_pairs = []

        if replace_pairs:
            # Long words first, avoid "short words hitting inside long words"
            replace_pairs.sort(key=lambda p: len(p[0]), reverse=True)

            # Construct pattern with optional "<color> mask " prefix: only replace core org, don't damage prefix
            bl = r"\b" if word_boundary else ""
            br = r"\b" if word_boundary else ""
            prefix_pat = r"(?P<prefix>\b[A-Za-z]+\s+mask\s+)?"  # Optional prefix group

            def _build_pat(org: str) -> re.Pattern:
                return re.compile(rf"{prefix_pat}({bl}{re.escape(org)}{br})", flags)

            def _apply_replacements_to_string(s: str) -> Tuple[str, List[Tuple[str, str, int]]]:
                rops: List[Tuple[str, str, int]] = []
                for org, repl in replace_pairs:
                    pat = _build_pat(org)
                    def _sub_fn(m):
                        pref = m.group("prefix") or ""
                        return pref + repl
                    s, cnt = pat.subn(_sub_fn, s)
                    if cnt > 0:
                        rops.append((org, repl, cnt))
                return s, rops

            replaced_ops: List[Tuple[str, str, int]] = []

            if isinstance(result, str):
                result, rops = _apply_replacements_to_string(result)
                replaced_ops.extend(rops)
            elif isinstance(result, list):
                new_list2: List[str] = []
                for i, elem in enumerate(result):
                    if isinstance(elem, str):
                        new_s, rops = _apply_replacements_to_string(elem)
                        replaced_ops.extend(rops)
                        new_list2.append(new_s)
                    else:
                        new_list2.append(elem)
                result = new_list2

            # Append statistics to report
            report["replaced"] = replaced_ops
            report["num_replaced"] = sum(cnt for _, _, cnt in replaced_ops)

    return result, report

def load_color_json(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Read color.json and return a dictionary containing two lists: "objs" and "loca".
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "objs" not in data or "loca" not in data:
        raise ValueError("color.json must contain 'objs' and 'loca'")
    return data

def flatten_strlist1(nested):
    """Flatten any nested task list into a one-dimensional string list (filter out blanks)."""
    from collections.abc import Iterable
    flat = []
    for x in (nested or []):
        if isinstance(x, str):
            s = x.strip()
            if s:
                flat.append(s)
        elif isinstance(x, Iterable):
            flat.extend(flatten_strlist1(x))
        else:
            flat.append(str(x))
    return flat

def make_instruction(subtask, tasklist, sep=", ",
                     template="now do {sub}, the whole task is {full}",simple=False):
    """Return string version of prompt."""
    if not simple:
        tl = flatten_strlist1(tasklist)
        full = sep.join(tl) if tl else "the task"
        sub = subtask.strip() if isinstance(subtask, str) else " ".join(flatten_strlist1(subtask))
        task = template.format(sub=sub, full=full)
    else:
        task = ', '.join(tasklist)
    return task

def delete_color(tasklist: list[str]) -> list[str]:
    """
    Remove all color+mask prompt words from each string in the list
    
    For example:
    "red mask apple" -> "apple"
    "blue mask banana green mask orange" -> "banana orange"
    "mask apple yellow mask banana" -> "apple banana"
    """
    tasklist_nocolor = []
    
    for task in tasklist:
        words = task.split()
        to_remove = set()  # Use set to store indices to be removed
        
        # Find positions of all "mask" and their corresponding color words
        for i, word in enumerate(words):
            if word.lower() == "mask":
                to_remove.add(i)  # Add mask index
                if i > 0:
                    to_remove.add(i-1)  # Add index of word before mask
        
        # Keep words not marked for deletion
        filtered_words = [word for i, word in enumerate(words) if i not in to_remove]
        tasklist_nocolor.append(" ".join(filtered_words))
    
    return tasklist_nocolor

# ==== Anti-stuck: action-only validator ======================================
import numpy as _np
from collections import deque as _deque

def check_stuck_by_actions(
    action_hist: "_deque[_np.ndarray]",
    *,
    min_len: int = 8,
    trans_thresh: float = 0.02,   # translation magnitude threshold (same units as your normalized actions)
    rot_thresh: float = 0.03,     # rotation magnitude threshold
    grip_change_thresh: float = 0.1,  # gripper channel (7th dim) change threshold (≈0 for binary -1/1)
    oscillation_flip_ratio: float = 0.6  # oscillation criterion: fraction of sign flips threshold
) -> tuple[bool, str]:
    """
    Stuck detection relying only on action history:
      Condition 1 (weak motion): recent min_len steps have very small translation and rotation magnitudes,
         and the gripper is nearly unchanged;
      Condition 2 (oscillation): recent min_len steps show frequent sign flips on translation/rotation axes
         and means close to zero.
    Returns: (is_stuck, reason)
    """
    n = len(action_hist)
    if n < min_len:
        return False, f"history too short ({n}<{min_len})"

    # Take the most recent min_len steps
    recent = list(action_hist)[-min_len:]
    A = _np.stack([_np.asarray(a, dtype=float).reshape(-1) for a in recent], axis=0)  # [T,7]

    if A.shape[1] < 7:
        # Safety check: actions should be 7-dimensional [tx,ty,tz, rx,ry,rz, gripper]
        return False, f"invalid action dim {A.shape[1]}"

    trans = A[:, 0:3]
    rot   = A[:, 3:6]
    grip  = A[:, 6]

    trans_mag = _np.linalg.norm(trans, axis=1)      # [T]
    rot_mag   = _np.linalg.norm(rot, axis=1)        # [T]
    grip_delta= _np.abs(_np.diff(grip)).max(initial=0.0)

    # Condition 1: sustained weak motion & gripper almost static
    cond_weak_motion = (trans_mag < trans_thresh).all() and (rot_mag < rot_thresh).all()
    cond_grip_static = (grip_delta <= grip_change_thresh)
    if cond_weak_motion and cond_grip_static:
        return True, "weak-motion-with-static-gripper"

    # Condition 2: oscillation (frequent sign flips and mean approx 0)
    def _flip_ratio(arr2d: _np.ndarray) -> float:
        # Compute per-axis sign flip ratio, then average across axes
        signs = _np.sign(arr2d)  # [-1,0,1]
        flips = _np.sum(_np.abs(_np.diff(signs, axis=0)) == 2, axis=0)  # count -1<->1 flips only
        return float(_np.mean(flips / max(1, signs.shape[0]-1)))

    tr_flip_r = _flip_ratio(trans)
    rr_flip_r = _flip_ratio(rot)
    mean_small = (_np.linalg.norm(trans.mean(0)) < trans_thresh) and (_np.linalg.norm(rot.mean(0)) < rot_thresh)

    if (tr_flip_r > oscillation_flip_ratio or rr_flip_r > oscillation_flip_ratio) and mean_small:
        return True, f"oscillation(tr={tr_flip_r:.2f}, rr={rr_flip_r:.2f})"

    return False, "ok"

