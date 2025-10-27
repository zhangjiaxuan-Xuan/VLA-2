import nltk,re,os
from typing import Any, Dict, List, Tuple, Union, Optional, Iterable
import collections
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import json
from PIL import Image
def decompose_liberotask(task: str) -> Dict[str, List[str]]:
    tokens = word_tokenize(task)
    tagged = pos_tag(tokens)
    the_idxs = [i for i, tok in enumerate(tokens) if tok.lower() == "the"]
    defs = []
    for idx in (the_idxs[:1] + the_idxs[-1:])[:2]:
        np_words = []
        for tok, tag in tagged[idx+1:]:
            if tag.startswith("JJ") or tag.startswith("NN"):
                np_words.append(tok)
            else:
                break
        if np_words:
            defs.append(" ".join(np_words))
    return {"tasklist": [task], "objs": defs}

def enrich_subtask_v2(
    subtask: str,
    media: Union[
        Tuple[Union[str, Image.Image], Union[str, Image.Image]],  # Real video paths or PIL Images
        Dict[str, List[Union[str, Image.Image]]],                 # {"main_flow":[...], "wrist_flow":[...]} or {"flow":[...]}
        collections.deque,                                        # deque([...]): supports single flow or (main, wrist) pairs
        List[Tuple[Union[str, Image.Image], Union[str, Image.Image]]],  # [(main_i, wrist_i), ...]
        List[Union[str, Image.Image]],                            # [frame1, frame2, ...]
    ],
    *,
    input_mode: Optional[str] = None,    # "video" / "image_flow"; auto-detect if None
    max_frames: Optional[int] = None,    # image_flow: keep only the latest N frames
    stride: int = 1,                     # image_flow: frame sampling stride
    flow_order: str = "old_to_new",      # or "new_to_old"
    title_prefix: str = "",              # optional prefix for prompt text
    tmp_dir: Optional[str] = "./experiments/cache/image_flow",       # directory to store PIL.Image frames temporarily
    pil_format: str = "JPEG",            # save format for PIL images
    pil_quality: int = 90                # JPEG quality if format is JPEG
) -> List[Dict[str, Any]]:
    """
    v3+: Qwen2.5-VL compliant:
      - Real videos: {"type":"video","video": "/path/video.mp4"}
      - Multi-frame sequence: {"type":"video","video": ["/path/f1.jpg", "/path/f2.jpg", ...]}
    NLTK-based parsing is prioritized for extracting objects/locations from "place" tasks;
    falls back to simple logic if NLTK fails.
    Supports deque/list/dict inputs; if frames are PIL.Image, saves them to tmp_dir.
    """

    # ===== 0) Utility: save PIL.Image to tmp_dir and return file path =====
    _counter = {"v": 0}  # simple counter to avoid filename collisions
    def _ensure_tmp_path(obj: Union[str, Image.Image], prefix: str) -> str:
        """
        Return a file URI like 'file:///path/to/frame.jpg' for either a path-like string
        or a PIL.Image (which will be saved into tmp_dir first).
        """
        from pathlib import Path
        def _to_file_uri(p: Union[str, Path]) -> str:
            p = Path(str(p)).expanduser().resolve()        # absolute, canonical
            return p.as_uri()                              # -> 'file:///...'

        # 1) Path-like input
        if isinstance(obj, (str, bytes)):
            s = obj.decode() if isinstance(obj, bytes) else obj
            # If it's already a URI, pass through
            if isinstance(s, str) and (s.startswith("file://") or s.startswith("http://") or s.startswith("https://")):
                return s
            return _to_file_uri(s)

        # 2) PIL.Image input -> save then return file URI
        if not isinstance(obj, Image.Image):
            raise ValueError("Frame element is neither a path string/bytes nor a PIL.Image.")

        if not tmp_dir:
            raise ValueError("PIL.Image detected but tmp_dir not provided.")
        os.makedirs(tmp_dir, exist_ok=True)

        _counter["v"] += 1
        ext = "jpg" if pil_format.upper() == "JPEG" else pil_format.lower()
        fname = f"{prefix}_{_counter['v']:06d}.{ext}"
        fpath = Path(tmp_dir) / fname

        save_params = {}
        if pil_format.upper() == "JPEG":
            save_params.update(dict(quality=pil_quality, optimize=True))
        obj.save(str(fpath), pil_format, **save_params)

        return _to_file_uri(fpath)

    # ===== 1) Extract objects and locations =====
    try:
        info = decompose_liberotask(subtask)
        objs = info.get("objs", [])
    except Exception:
        objs = []
    object_name = objs[0] if len(objs) >= 1 else "the object"
    location_name = objs[1] if len(objs) >= 2 else "the target location"

    # ===== 2) Identify verb and extract raw_part =====
    inst_low = subtask.lower()
    verb, raw_part = "", ""
    for v in ("pick up", "place", "turn on", "turn off", "open", "close"):
        if inst_low.startswith(v):
            verb = v
            try:
                m = re.search(rf'(?i){re.escape(v)}\s*(.*)', subtask)
                raw_part = m.group(1).rstrip('.') if m else ""
            except Exception:
                raw_part = ""
            break

    # ===== 3) NLTK-priority parsing for "place", fallback to simple rules =====
    def _nltk_place_parse(raw: str) -> Tuple[str, str]:
        from nltk import word_tokenize, pos_tag
        tokens = word_tokenize(raw)
        tagged = pos_tag(tokens)
        object_tokens, location_tokens, hit_prep = [], [], False
        for word, tag in tagged:
            if not hit_prep and (tag.startswith("JJ") or tag.startswith("NN") or tag.startswith("DT")):
                object_tokens.append(word)
            elif tag == "IN" or hit_prep:
                hit_prep = True
                location_tokens.append(word)
        obj = " ".join(object_tokens).strip() if object_tokens else raw.strip()
        loc = " ".join(location_tokens).strip()
        return obj, loc

    def _fallback_place_parse(raw: str) -> Tuple[str, str]:
        preps = (" in ", " on ", " into ", " onto ", " at ", " near ")
        for p in preps:
            if p in f" {raw} ":
                o, l = raw.split(p, 1)
                obj = o.strip() or object_name
                loc = (p.strip() + " " + l.strip()) if l.strip() else location_name
                return obj, loc
        return (raw.strip() or object_name, location_name)

    if verb == "place" and raw_part:
        try:
            obj_nltk, loc_nltk = _nltk_place_parse(raw_part)
            if obj_nltk: object_name = obj_nltk
            if loc_nltk: location_name = loc_nltk
        except Exception:
            obj_fb, loc_fb = _fallback_place_parse(raw_part)
            object_name, location_name = obj_fb, loc_fb
    else:
        if not object_name and raw_part:
            for sep in (" in ", " on ", " into ", " onto ", " at ", " near "):
                if sep in raw_part:
                    o, l = raw_part.split(sep, 1)
                    if o.strip(): object_name = o.strip()
                    if l.strip(): location_name = l.strip()
                    break

    # ===== 4) Build prompt text =====
    prefix = (
        f"{title_prefix + ' - ' if title_prefix else ''}"
        f"Observe the inputs (two videos or two image-flow videos). "
        f"The subtask robot arm is currently working on: '{subtask}'. "
    )
    if verb == "pick up":
        prompt = (
            f"{prefix} Based *Only* on the provided media, has '{object_name}' or anything else been grasped and lifted off any surface by the end? "
            "Answer 'Yes' or 'No'."
        )
    elif verb == "place":
        prompt = (
            f"{prefix} Based *Only* on the provided media, has '{object_name}' or anything else been placed '{location_name}' and is the gripper away? "
            "Answer 'Yes' or 'No'."
        )
    elif verb in ("turn on", "turn off", "open", "close"):
        target = raw_part or object_name
        action_text = {
            "turn on": "turned on (powered up)",
            "turn off": "turned off (powered down)",
            "open": "fully opened",
            "close": "fully closed",
        }[verb]
        prompt = (
            f"{prefix} Based *Only* on the provided media, has '{target}' or anything else been {action_text} by the end? "
            "Answer 'Yes' or 'No'."
        )
    else:
        prompt = (
            f"{prefix} Based *Only* on the provided media, has the instructed action completed successfully by the end? "
            "Answer 'Yes' or 'No'."
        )

    # ===== 5) Detect input mode =====
    mode = input_mode
    if mode is None:
        if isinstance(media, tuple) and len(media) == 2 and all(isinstance(x, (str, Image.Image, bytes)) for x in media):
            mode = "video"
        else:
            mode = "image_flow"

    # ===== 6) Helpers =====
    def _as_list(seq: Iterable) -> List:
        return list(seq) if not isinstance(seq, (str, bytes)) else [seq]

    def _apply_sampling(flow_list: List[str]) -> List[str]:
        xs = flow_list
        if max_frames is not None and max_frames > 0:
            xs = xs[-max_frames:]  # keep the latest N frames (old→new order)
        if stride and stride > 1:
            xs = xs[::stride]
        return xs

    def _maybe_reverse(xs: List[str]) -> List[str]:
        return xs if flow_order == "old_to_new" else list(reversed(xs))

    # ===== 7) Build content according to Qwen image-flow-as-video spec =====
    content: List[Dict[str, Any]] = []

    if mode == "video":
        # Real video mode
        if not (isinstance(media, tuple) and len(media) == 2):
            raise ValueError("In video mode, media must be (main_video_path, wrist_video_path).")
        main_path = _ensure_tmp_path(media[0], "main_video")
        wrist_path = _ensure_tmp_path(media[1], "wrist_video")
        content.extend([
            {"type": "text", "text": "Video 1 (Main View):"},
            {"type": "video", "video": str(main_path),"fps":2.0},
            {"type": "text", "text": "Video 2 (Wrist View):"},
            {"type": "video", "video": str(wrist_path),"fps":2.0},
        ])

    else:
        # Image flow mode
        main_flow: List[str] = []
        wrist_flow: List[str] = []
        single_flow: List[str] = []

        if isinstance(media, dict):
            if "main_flow" in media or "wrist_flow" in media:
                main_flow = [_ensure_tmp_path(x, "main") for x in _as_list(media.get("main_flow", []))]
                wrist_flow = [_ensure_tmp_path(x, "wrist") for x in _as_list(media.get("wrist_flow", []))]
            elif "flow" in media:
                single_flow = [_ensure_tmp_path(x, "flow") for x in _as_list(media["flow"])]
            else:
                raise ValueError("In image_flow mode, dict must contain 'main_flow'/'wrist_flow' or 'flow'.")

        elif isinstance(media, (collections.deque, list, tuple)):
            if len(media) == 0:
                single_flow = []
            else:
                first = media[0]
                # Paired frames: [(main_i, wrist_i), ...]
                if isinstance(first, (list, tuple)) and len(first) == 2:
                    for pair in media:
                        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                            raise ValueError("Detected paired frames but an element length != 2.")
                        m, w = pair
                        m_path = _ensure_tmp_path(m, "main")
                        w_path = _ensure_tmp_path(w, "wrist")
                        main_flow.append(m_path)
                        wrist_flow.append(w_path)
                # Single flow: [img1, img2, ...]
                else:
                    for it in media:
                        single_flow.append(_ensure_tmp_path(it, "flow"))
        else:
            raise ValueError("In image_flow mode, media must be dict / deque / list / tuple.")

        # Apply sampling and ordering
        if main_flow or wrist_flow:
            main_flow  = _maybe_reverse(_apply_sampling(main_flow))
            wrist_flow = _maybe_reverse(_apply_sampling(wrist_flow))
            if not main_flow or not wrist_flow:
                raise ValueError("Detected dual image flows but one is empty.")
            content.append({"type": "text", "text": "Video 1 (Main View, image flow):"})
            content.append({"type": "video", "video": main_flow,"fps":2.0})
            content.append({"type": "text", "text": "Video 2 (Wrist View, image flow):"})
            content.append({"type": "video", "video": wrist_flow,"fps":2.0})
        else:
            single_flow = _maybe_reverse(_apply_sampling(single_flow))
            if not single_flow:
                raise ValueError("In image_flow mode, at least one frame sequence must be non-empty.")
            content.append({"type": "text", "text": "Video (Image Flow):"})
            content.append({"type": "video", "video": single_flow,"fps":2.0})

    # Append final question
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]

def enrich_subtask(
    subtask: str,
    video_pair: Tuple[str, str]  # (main_view_path, wrist_view_path)
) -> List[Dict[str, Any]]:
    """
    Semantic enhancement for a single subtask:
      - subtask: subtask instruction text
      - video_pair: (main_video_path, wrist_video_path)
    Returns:
      messages: [{"role": "user", "content": [...]}]
    """
    # —— I. Extract objects and locations —— #
    try:
        info = decompose_liberotask(subtask)
        objs = info.get("objs", [])
    except Exception:
        objs = []
    object_name = objs[0] if len(objs) >= 1 else "the object"
    location_name = objs[1] if len(objs) >= 2 else "the target location"

    # —— II. Identify verbs & split original parts —— #
    inst_low = subtask.lower()
    verb = ""
    raw_part = ""
    for v in ("pick up", "place", "turn on", "turn off", "open", "close"):
        if inst_low.startswith(v):
            verb = v
            try:
                m = re.search(rf'(?i){re.escape(v)}\s*(.*)', subtask)
                raw_part = m.group(1).rstrip('.') if m else ""
            except Exception:
                raw_part = ""
            break

    # Extract separation for place and update object_name/location_name
    if verb == "place" and raw_part:
        # Tokenization + POS tagging
        tokens = word_tokenize(raw_part)
        tagged = pos_tag(tokens)

        object_tokens = []
        location_tokens = []
        hit_prep = False

        # Scan word by word
        for word, tag in tagged:
            if not hit_prep and (tag.startswith("JJ") or tag.startswith("NN") or tag .startswith("DT")):
                # Before encountering the first IN, collect adjectives/nouns
                object_tokens.append(word)
            else:
                # Once tag == 'IN', start collecting for location_name (including this preposition)
                hit_prep = True
                location_tokens.append(word)

        # If completely no NN/JJ encountered, fall back to entire raw_part  
        object_name = " ".join(object_tokens) if object_tokens else raw_part.strip()
        # If no location_tokens collected, leave empty
        location_name = " ".join(location_tokens).strip()

    else:
        # Handle other verb branches or failures
        prep_str = ''

    if not object_name:
        for sep in (" in ", " on ", " into ", " onto ","at" ,"near"):
            if sep in raw_part:
                o, l = raw_part.split(sep, 1)
                if o.strip(): object_name = o.strip()
                if l.strip(): location_name = l.strip()
                break

    # —— III. Concatenate prefix and prompt —— #
    prefix = (
        f"Observe 2 videos (main view first, then wrist view). "
        f"The subtask robot arm currently working on is: '{subtask}'. "
    )

    if verb == "pick up":
        prompt = (
            f"{prefix} Based *Only* on the video pairs. Has '{object_name}' been grasped and lifted off any surface by end of videos? "
            "Answer 'Yes' or 'No'."
        )
    elif verb == "place":
        prompt = (
            f"{prefix} Based *Only* on the video pairs. Has '{object_name}' been placed '{location_name}' and is the gripper away? "
            "Answer 'Yes' or 'No'."
        )
    elif verb == "turn on":
        target = raw_part or object_name
        prompt = (
            f"{prefix} Based *Only* on the video pairs. Has '{target}' been turned on (powered up) by end of videos? "
            "Answer 'Yes' or 'No'."
        )
    elif verb == "turn off":
        target = raw_part or object_name
        prompt = (
            f"{prefix} Based *Only* on the video pairs. Has '{target}' been turned off (powered down) by end of videos? "
            "Answer 'Yes' or 'No'."
        )
    elif verb == "open":
        target = raw_part or object_name
        prompt = (
            f"{prefix} Based *Only* on the video pairs. Has '{target}' been fully opened by end of videos? "
            "Answer 'Yes' or 'No'."
        )
    elif verb == "close":
        target = raw_part or object_name
        prompt = (
            f"{prefix} Based *Only* on the video pairs. Has '{target}' been fully closed by end of videos? "
            "Answer 'Yes' or 'No'."
        )
    else:
        prompt = (
            f"{prefix} Based *Only* on the video pairs. Has the instructed action completed successfully by end of videos? "
            "Answer 'Yes' or 'No'."
        )

    # —— IV. Construct content list —— #
    main_path, wrist_path = video_pair
    content = [
        {"type": "text", "text": "Video 1 (Main View):"},
        {"type": "video", "video": main_path},
        {"type": "text", "text": "Video 2 (Wrist View):"},
        {"type": "video", "video": wrist_path},
        {"type": "text", "text": prompt}
    ]

    return [{"role": "user", "content": content}]

def main():
    # Example subtasks to test
    subtasks = [
        "Pick up the red block from the table.",
        "Place the Red mask green mug into the box near the blue cabinet.",
        "Turn on the lamp.",
        "Open the door.",
        "Turn off the fan.",
        "Some custom action without keywords."
    ]

    # Example video paths for testing
    video_pair: Tuple[str, str] = ("videos/main_view.mp4", "videos/wrist_view.mp4")

    for subtask in subtasks:
        print(f"\n=== Testing subtask: {subtask} ===")
        try:
            # Test decompose_liberotask directly
            objs = decompose_liberotask(subtask).get("objs", [])
            print("Extracted objects:", objs)

            # Test enrich_subtask and pretty-print result
            messages = enrich_subtask(subtask, video_pair)
            print("Enriched messages:")
            print(json.dumps(messages, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Error processing '{subtask}': {e}")


if __name__ == "__main__":
    main()