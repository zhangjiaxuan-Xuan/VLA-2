import json
from pathlib import Path
import threading
import queue
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from ultralytics import SAM                              # SAM2.1-b
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
from hydra.core.global_hydra import GlobalHydra
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="cutie")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp")

# === User configuration ===
# Replace these paths with your repository-relative or absolute locations before running.
# For portability we leave placeholders; users can set concrete paths or override via env/config.
SRC_ROOT     = Path("path/to/your/boxdatas")    # Source data root (Path-like)
DEST_ROOT    = Path("path/to/your/seg-video")   # Output root (Path-like)
SAM_WEIGHTS  = "path/to/your/sam2.1_l.pt"       # Path to SAM weights file
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MAX_INTERNAL = -1
ALPHA        = 0.30                               # Overlay alpha for masks
# Strength parameter used when adjusting color brightness
HSV_STRENGTH = 0.5
# =====================


# ========== Three convenient external interface functions ==========

def _ensure_pil_palette(palette: list[int]) -> list[int]:
    """
    Ensure the length of the palette passed to PIL is valid (768 = 256 * 3). 
    If it's too short, pad with zeros; if it's too long, truncate it.
    PIL's putpalette requires a flat [r, g, b, ...] list of length 768.

    Ensure the palette length passed to PIL is valid (768 = 256 * 3).
    If it's too short, pad with zeros; if it's too long, truncate it.
    PIL's putpalette requires a flat [r, g, b, ...] list of length 768.
    """
    pal = palette[:] if isinstance(palette, list) else list(palette)
    # Pad length to a multiple of 3
    if len(pal) % 3 != 0:
        pal += [0] * (3 - (len(pal) % 3))
    # Pad to 768
    if len(pal) < 768:
        pal += [0] * (768 - len(pal))
    return pal[:768]


def init_models_and_params(
    sam_weights: str = SAM_WEIGHTS,
    device: str = DEVICE,
    max_internal: int = MAX_INTERNAL,
):
    """
        Function 1: Initialize all models and parameters (without performing the first-frame step).
        Returns:
            sam_model: Ultralytics SAM model (moved to device and eval mode)
            processor: Cutie InferenceCore (not yet initialized with the first frame)
            device:    Current device string
    """
    # Note: Don't clear GlobalHydra here, leave it to function 2
    sam_model = SAM(sam_weights).to(device).eval()
    cutie_net = get_default_model().to(device).eval()
    processor = InferenceCore(cutie_net, cfg=cutie_net.cfg)
    processor.max_internal_size = max_internal
    return sam_model, processor, device


def init_processor_first_step(
    sam_model: SAM,
    processor: InferenceCore,
    first_frame: np.ndarray,
    *,
    # Three initialization methods (choose one): provide label_mask+objects; or provide dirt; or provide box_data
    label_mask: np.ndarray | None = None,
    objects: list[int] | None = None,
    palette: list[int] | None = None,
    dirt: list[dict] | None = None,
    box_data: dict | None = None,
    device: str = DEVICE,
) -> tuple[InferenceCore, list[int], list[int]]:
    """
    Function 2: Initialize GlobalHydra and complete the first step of Cutie using "first frame + mask + object IDs".
    - You can directly provide label_mask (H, W uint8) and objects (ID list), or
    - Provide a dirt list (containing box/score/color), or
    - Provide a box_data structure ({"objs": {video: [bboxes...]}, "loca": {...}})
    This function will:
      1) GlobalHydra.instance().clear()
      2) If needed, use SAM to generate label_mask based on the first frame and boxes
      3) Assemble/complete the palette
      4) Call processor.step(first_t, mask_t, objects=objects)
    Returns:
      processor (initialized with the first frame)
      palette (PIL flat palette, including background black)
      objects (list of object IDs, corresponding to values in the mask)

        Function 2: Initialize GlobalHydra and complete Cutie's first step using
        (first frame + mask + object IDs).
        - You may provide label_mask (H, W uint8) and objects (list of IDs), or
        - provide a `dirt` list (each item contains box/score/color), or
        - provide a `box_data` structure ({"objs":{video:[bboxes...]}, "loca":{...}}).
        This function will:
            1) Clear GlobalHydra.instance()
            2) Optionally use SAM to generate label_mask from boxes
            3) Assemble / pad the palette
            4) Call processor.step(first_t, mask_t, objects=objects)
        Returns:
            processor (initialized with the first frame)
            palette (flat PIL palette including background black)
            objects (list of object IDs corresponding to values in the mask)
    """
    # 1) Reset Hydra
    GlobalHydra.instance().clear()
    cutie_net = get_default_model().to(device).eval()
    processor = InferenceCore(cutie_net, cfg=cutie_net.cfg)
    H, W = first_frame.shape[:2]

    # 2) Generate label_mask / objects / palette
    if palette is None:
        palette = [0, 0, 0]  # background black

    built_from_boxes = False

    if label_mask is None:
        label_mask = np.zeros((H, W), dtype=np.uint8)
        objects = [] if objects is None else objects
        next_id = 1 if len(objects) == 0 else (max(objects) + 1)

        if dirt is not None:
            # dirt: [{"box":[x1,y1,x2,y2] or [-1...], "score":float, "color":(r,g,b)}, ...]
            for entry in dirt:
                bbox = entry.get("box", [])
                if bbox == [-1, -1, -1, -1] or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                res = sam_model.predict(source=first_frame, device=device, bboxes=[[x1, y1, x2, y2]])
                if not res or res[0].masks is None:
                    continue
                mask_full = (res[0].masks.data[0].cpu().numpy() > 0.5)

                roi = mask_full[y1:y2, x1:x2]
                region = label_mask[y1:y2, x1:x2]
                write_idx = roi & (region == 0)
                region[write_idx] = next_id
                label_mask[y1:y2, x1:x2] = region

                base_rgb = entry.get("color", (255, 255, 255))
                score = float(entry.get("score", 0.5))
                adj_rgb = adjust_color_by_score(tuple(base_rgb), score)
                palette.extend(list(adj_rgb))

                objects.append(next_id)
                next_id += 1
            built_from_boxes = True

        elif box_data is not None:
            # box_data: {"objs":{video:[(x1,y1,x2,y2),...]}, "loca":{video:[...]} }
            for cls in ("objs", "loca"):
                for boxes in box_data.get(cls, {}).values():
                    for (x1, y1, x2, y2) in boxes:
                        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                        res = sam_model.predict(source=first_frame, device=device, bboxes=[[x1, y1, x2, y2]])
                        if not res or res[0].masks is None:
                            continue
                        mask_full = (res[0].masks.data[0].cpu().numpy() > 0.5)

                        roi = mask_full[y1:y2, x1:x2]
                        region = label_mask[y1:y2, x1:x2]
                        write_idx = roi & (region == 0)
                        region[write_idx] = next_id
                        label_mask[y1:y2, x1:x2] = region

                        # Assign color: prefer COLOR_MAP, else random
                        if cls == "objs":
                            base_rgb = COLOR_MAP.get("blue", (255, 0, 0))
                        else:
                            base_rgb = COLOR_MAP.get("red", (0, 0, 255))
                        palette.extend(list(base_rgb))
                        objects.append(next_id)
                        next_id += 1
            built_from_boxes = True

        else:
            # No boxes and no mask provided; keep empty mask and objects (not recommended)
            pass

        if len(objects) == 0 and built_from_boxes:
            # Defensive: boxes provided but no objects written
            uniq = np.unique(label_mask)
            objects = [int(v) for v in uniq if v != 0]

    # If user provided label_mask but no objects, infer objects from the mask
    if objects is None or len(objects) == 0:
        uniq = np.unique(label_mask)
        objects = [int(v) for v in uniq if v != 0]

    # Pad palette into the flat format required by PIL
    palette = _ensure_pil_palette(palette)

    # 3) Cutie first-step initialization
    first_t = to_tensor(Image.fromarray(first_frame)).to(device).float()
    mask_t = torch.from_numpy(label_mask).to(device)
    processor.step(first_t, mask_t, objects=objects)

    return processor, palette, objects


def track_frame_and_overlay(
    frame: np.ndarray,
    processor: InferenceCore,
    palette: list[int],
    *,
    device: str = DEVICE,
    alpha: float = ALPHA
) -> np.ndarray:
    """
    Function 3: Input the current frame and output the image with overlaid tracking masks (BGR).
    Reuses the internal segment_and_track function but ensures the palette is safely padded first.

    Function 3: Input the current frame and output the image with overlaid tracking masks (BGR).
    Reuses the internal segment_and_track function, but ensures the palette is safely padded first.
    """
    # segment_and_track global ALPHA；here allow external temporary override
    global ALPHA
    old_alpha = ALPHA
    ALPHA = alpha
    try:
        pal = _ensure_pil_palette(palette)
        out = segment_and_track(frame, processor, pal)  # return uint8 BGR
    finally:
        ALPHA = old_alpha
    return out
# ========== End of three interface functions ==========


def reset_processor_memory(processor: InferenceCore):
    """Clear Cutie internal memory buffers."""
    processor.memory.clear_sensory_memory()      # clear image_feature_store
    processor.memory.clear_non_permanent_memory()

COLOR_MAP = {
    "red":  (0, 0, 255),
    "blue":   (255, 0, 0),
    "green": (0, 255, 0),
}

@torch.inference_mode()
def segment_and_track(frame, processor, palette):
    img_t = to_tensor(Image.fromarray(frame)).to(DEVICE).float()
    with torch.no_grad():
        output_prob = processor.step(img_t)
    mask = processor.output_prob_to_mask(output_prob)
    if mask.ndim == 3: mask = mask[0]
    mask = mask.detach().clone()  
    label = mask.cpu().numpy().astype(np.uint8)

    mask_pil = Image.fromarray(label)
    mask_pil.putpalette(palette)
    mask_color = np.array(mask_pil.convert("RGB"), dtype=np.uint8)

    out = frame.astype(np.float32)
    nz  = label > 0
    out[nz] = out[nz] * (1-ALPHA) + mask_color[nz] * ALPHA
    return out.astype(np.uint8)

def process_one_video(
    video_path: Path,
    box_data: dict,
    output_path: Path,
    sam_model: SAM,
    processor: InferenceCore,
    base_palette: list[int]
):
    """Use initialized sam_model and processor to perform segmentation tracking on a single video and save"""
    cap = cv2.VideoCapture(str(video_path))
    ret, first = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Unable to read video {video_path}")

    H, W = first.shape[:2]
    label_mask = np.zeros((H, W), dtype=np.uint8)
    palette    = base_palette.copy()
    obj_id     = 1
    objects    = []

    # SAM generates initial label_mask according to box_data
    for cls in ("objs", "loca"):
        for boxes in box_data.get(cls, {}).values():
            for (x1, y1, x2, y2) in boxes:
                res       = sam_model.predict(source=first, device=DEVICE,
                                              bboxes=[[x1, y1, x2, y2]])
                mask_full = (res[0].masks.data[0].cpu().numpy() > 0.5)
                roi       = mask_full[y1:y2, x1:x2]
                region    = label_mask[y1:y2, x1:x2]
                write_idx = (roi & (region == 0))
                region[write_idx] = obj_id
                label_mask[y1:y2, x1:x2] = region
                # Assign color to this object_id
                palette.extend(np.random.randint(0, 255, 3).tolist())
                objects.append(obj_id)
                obj_id += 1

    # Initialize Cutie with the first frame and generated label_mask
    first_t = to_tensor(Image.fromarray(first)).to(DEVICE).float()
    mask_t  = torch.from_numpy(label_mask).to(DEVICE)
    processor.step(first_t, mask_t, objects=objects)

    # Video writing configuration
    fps    = cap.get(cv2.CAP_PROP_FPS) or 10
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

    # Reset to first frame and process frame by frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out = segment_and_track(frame, processor, palette)
        writer.write(out)

    cap.release()
    writer.release()

def adjust_color_by_score(rgb_color: tuple[int,int,int], score: float) -> tuple[int,int,int]:
    """
    Adjust the brightness of rgb_color based on score [0,1]:
      - score > 0.5 → darker color
      - score < 0.5 → lighter color
    Use HSV color space to perform linear mapping on the V channel.
    """
    # 1. Normalize to [0,1]
    r, g, b = [c / 255.0 for c in rgb_color]
    # 2. Convert to HSV
    hsv = cv2.cvtColor(
        np.array([[[r, g, b]]], dtype=np.float32),
        cv2.COLOR_RGB2HSV
    )[0,0]
    h, s, v = hsv
    # 3. Calculate brightness factor: score=1→v*(1-0.5*strength); score=0→v*(1+0.5*strength)
    v_factor = 1.0 - (score - 0.5) * HSV_STRENGTH * 2
    v_factor = float(np.clip(v_factor, 0.0, 2.0))
    # 4. Apply and convert back to RGB
    hsv_adj = np.array([[[h, s, v * v_factor]]], dtype=np.float32)
    rgb_adj = cv2.cvtColor(hsv_adj, cv2.COLOR_HSV2RGB)[0,0]
    return tuple((rgb_adj * 255).astype(np.uint8))

def _video_writer_thread(path, size, fps, frame_queue):
    """Background thread: consume queue and write video file"""
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    while True:
        frame = frame_queue.get()
        if frame is None:  # end signal
            break
        writer.write(frame)
    writer.release()

import os

def process_with_dirt_from_frames_v2(
    frames: list[np.ndarray],
    dirt: list[dict],
    out_dir: Path,
    out_name: str,
) -> Path:
    """
    Generate masked video from existing frames and dirt information.

    Optimization: After Cutie output, batch calculate color overlay for all frames on GPU,
    then use background thread for unified writing, avoiding synchronous I/O and multiple CPU↔GPU copies.
    """
    # 1. Prepare output directory and path
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / out_name
    fps = 10
    H, W = frames[0].shape[:2]

    # 2. Start video writing thread
    frame_queue = queue.Queue(maxsize=16)
    writer_thr = threading.Thread(
        target=_video_writer_thread,
        args=(str(out_path), (W, H), fps, frame_queue),
        daemon=True
    )
    writer_thr.start()

    # 3. Initialize SAM + Cutie
    GlobalHydra.instance().clear()
    sam_model = SAM(SAM_WEIGHTS).to(DEVICE).eval()
    cutie_net = get_default_model().to(DEVICE).eval()
    processor = InferenceCore(cutie_net, cfg=cutie_net.cfg)
    processor.max_internal_size = MAX_INTERNAL

    # 4. Use first frame to generate initial label_mask & palette
    first = frames[0]
    label_mask = np.zeros((H, W), dtype=np.uint8)
    palette = [[0, 0, 0]]  # background color list
    objects = []

    for entry in dirt:
        bbox = entry.get("box", [])
        if bbox == [-1, -1, -1, -1] or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        # SAM prediction mask
        res = sam_model.predict(source=first, device=DEVICE, bboxes=[[x1, y1, x2, y2]])
        mask_full = (res[0].masks.data[0].cpu().numpy() > 0.5)
        roi = mask_full[y1:y2, x1:x2]
        region = label_mask[y1:y2, x1:x2]
        write_idx = roi & (region == 0)
        region[write_idx] = len(palette)
        label_mask[y1:y2, x1:x2] = region

        # Adjust color and add to palette
        adj_rgb = adjust_color_by_score(entry.get("color"), entry.get("score"))
        if isinstance(adj_rgb, (list, tuple)) and len(adj_rgb) == 3:
            palette.append(list(adj_rgb))
            objects.append(len(palette) - 1)

    # 5. Initialize Cutie tracking with first frame
    first_t = to_tensor(Image.fromarray(first)).to(DEVICE).float()
    mask_t  = torch.from_numpy(label_mask).to(DEVICE).long()
    processor.step(first_t, mask_t, objects=objects)

    # 6. Prepare palette tensor
    flat = [c for sub in palette for c in sub]
    pal_t = torch.tensor(flat, dtype=torch.float32, device=DEVICE).view(-1, 3) / 255.0

    # 7. Batch calculate overlay results for all frames
    frames_out = []
    with torch.no_grad():
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to RGB
            img_t = to_tensor(Image.fromarray(frame)).to(DEVICE).float()
            prob = processor.step(img_t)
            m = processor.output_prob_to_mask(prob)
            if m.ndim == 3:
                m = m[0]
            # GPU coloring & alpha blend
            color_map = pal_t[m]              # (H,W,3)
            orig_rgb  = img_t.permute(1, 2, 0) # (H,W,3)
            blended = orig_rgb.clone()
            mask_bool = m > 0
            blended[mask_bool] = (
                orig_rgb[mask_bool] * (1 - ALPHA)
                + color_map[mask_bool] * ALPHA
            )

            # Convert back to CPU np array
            frame_rgb = (blended * 255).byte().cpu().numpy()  # RGB
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frames_out.append(frame_bgr)

    # 8. Write all frames uniformly
    for f in frames_out:
        frame_queue.put(f)
    frame_queue.put(None)
    writer_thr.join()

    return out_path

def process_with_dirt_from_frames_v3(
    frames: list[np.ndarray],
    dirt: list[dict],
    out_dir: Path,
    out_name: str,
) -> Path:
    """
    Generate masked video from existing frames and dirt information.

    Corrections:
      - Only assign ID/color to an object when pixels are actually written in the first frame (write_idx.any()==True), avoiding ID and palette "misalignment".
      - Remove color transformation based on score, strictly use entry['color'].
      - Add bbox clipping and robustness checks to avoid out-of-bounds and false objects caused by empty masks.

    Optimization: After Cutie output, batch calculate color overlay for all frames on GPU,
                  then use background thread for unified writing, avoiding synchronous I/O and multiple CPU↔GPU copies.
    """
    # import os, queue, threading
    # import cv2
    # import torch
    # import numpy as np
    # from PIL import Image
    # from torchvision.transforms.functional import to_tensor

    # 1. prepare output directory and path
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / out_name
    fps = 10
    H, W = frames[0].shape[:2]

    # 2. start video writing thread
    frame_queue = queue.Queue(maxsize=16)
    writer_thr = threading.Thread(
        target=_video_writer_thread,
        args=(str(out_path), (W, H), fps, frame_queue),
        daemon=True
    )
    writer_thr.start()

    # 3. init SAM + Cutie
    GlobalHydra.instance().clear()
    sam_model = SAM(SAM_WEIGHTS).to(DEVICE).eval()
    cutie_net = get_default_model().to(DEVICE).eval()
    processor = InferenceCore(cutie_net, cfg=cutie_net.cfg)
    processor.max_internal_size = MAX_INTERNAL

    # 4. first frame label_mask & palette
    first = frames[0]
    label_mask = np.zeros((H, W), dtype=np.uint8)

    # Note: Use "index as ID" palette layout here: index=0 is background
    palette: list[list[int]] = [[0, 0, 0]]  # index 0: background
    objects: list[int] = []

    for entry in dirt:
        bbox = entry.get("box", [])
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        if bbox == [-1, -1, -1, -1]:
            continue

        # Convert coordinates to int and clip to image boundaries
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        # SAM prediction mask (robustness check)
        res = sam_model.predict(source=first, device=DEVICE, bboxes=[[x1, y1, x2, y2]])
        if not res or getattr(res[0], "masks", None) is None:
            continue
        mdata = res[0].masks.data
        if mdata is None or mdata.shape[0] == 0:
            continue

        mask_full = (mdata[0].cpu().numpy() > 0.5)

        # Write within the clipped region
        roi = mask_full[y1:y2, x1:x2]
        region = label_mask[y1:y2, x1:x2]
        write_idx = roi & (region == 0)

        # ✅ Only assign new ID/append palette/add to objects when pixels are actually written
        if write_idx.any():
            new_id = len(palette)  # current position is the new ID to be assigned
            region[write_idx] = new_id
            label_mask[y1:y2, x1:x2] = region

            base_rgb = entry.get("color", (255, 255, 255))
            # Remove color transformation: strictly use base_rgb
            if isinstance(base_rgb, tuple):
                base_rgb = list(base_rgb)
            if not (isinstance(base_rgb, list) and len(base_rgb) == 3):
                base_rgb = [255, 255, 255]

            palette.append([int(base_rgb[0]), int(base_rgb[1]), int(base_rgb[2])])
            objects.append(new_id)
        # else: completely skip this entry, don't occupy ID and don't add to palette/objects

    # If necessary, do a fallback (generally won't trigger)
    if len(objects) == 0:
        uniq = np.unique(label_mask)
        objects = [int(v) for v in uniq if v != 0]

    # 5. Initialize Cutie tracking with first frame
    first_rgb = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
    first_t = to_tensor(Image.fromarray(first_rgb)).to(DEVICE).float()
    mask_t  = torch.from_numpy(label_mask).to(DEVICE).long()
    processor.step(first_t, mask_t, objects=objects)

    # 6. Prepare palette tensor (shape: [#ids, 3], range 0~1)
    flat = [c for sub in palette for c in sub]
    pal_t = torch.tensor(flat, dtype=torch.float32, device=DEVICE).view(-1, 3) / 255.0

    # 7. Batch calculate overlay results for all frames (coloring and alpha-blend on GPU)
    with torch.no_grad():
        for frame in frames:
            # Convert to RGB for inference and coloring
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_t = to_tensor(Image.fromarray(rgb)).to(DEVICE).float()

            prob = processor.step(img_t)
            m = processor.output_prob_to_mask(prob)
            # Ensure it's a 2D integer index
            if m.ndim == 3:
                m = m[0]
            m = m.long()

            # Color mapping & alpha blend (only for ID>0 regions)
            color_map = pal_t[m]                 # (H,W,3)
            orig_rgb  = img_t.permute(1, 2, 0)   # (H,W,3)
            blended = orig_rgb.clone()
            mask_bool = m > 0
            blended[mask_bool] = (
                orig_rgb[mask_bool] * (1 - ALPHA)
                + color_map[mask_bool] * ALPHA
            )

            # Return to CPU and convert to BGR for output
            frame_rgb = (blended * 255).byte().cpu().numpy()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_queue.put(frame_bgr)

    # 8. End writing
    frame_queue.put(None)
    writer_thr.join()

    return out_path

def process_with_dirt_from_frames(
    frames: list[np.ndarray],
    dirt: list[dict],
    out_dir: Path,
    out_name: str,
) -> Path:
    """
    generate masked video from existing frames and dirt information.

    Args:
      frames: List[np.ndarray], BGR image list
      dirt: List[{
        "prompt": str,
        "box": [x1,y1,x2,y2] or [-1,-1,-1,-1],
        "score": float,
        "color": (r,g,b)
      }]
      out_dir: Path output directory
      out_name: str output filename

    Returns:
      Path processed video file path
    """
    # Prepare output
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name

    # Reset Hydra configuration each time and initialize SAM + Cutie
    GlobalHydra.instance().clear()
    sam_model = SAM(SAM_WEIGHTS)
    cutie_net = get_default_model().to(DEVICE).eval()
    processor = InferenceCore(cutie_net, cfg=cutie_net.cfg)
    processor.max_internal_size = MAX_INTERNAL

    # Use first frame to generate initial label_mask and palette
    first = frames[0]
    H, W = first.shape[:2]
    label_mask = np.zeros((H, W), dtype=np.uint8)
    palette = [0, 0, 0]  # background black
    obj_id = 1
    objects = []

    # Build single-target mask, append brightness-adjusted colors to palette in order
    for entry in dirt:
        bbox = entry["box"]
        score = entry["score"]
        if bbox == [-1, -1, -1, -1]:
            continue
        x1, y1, x2, y2 = bbox
        # SAM predict target mask
        res = sam_model.predict(
            source=first, device=DEVICE, bboxes=[[x1,y1,x2,y2]]
        )
        mask_full = (res[0].masks.data[0].cpu().numpy() > 0.5)
        roi = mask_full[y1:y2, x1:x2]
        region = label_mask[y1:y2, x1:x2]
        write_idx = roi & (region == 0)
        region[write_idx] = obj_id
        label_mask[y1:y2, x1:x2] = region

        # Adjust color darker/lighter based on score
        base_rgb = entry["color"]
        adj_rgb  = adjust_color_by_score(base_rgb, score)
        palette.extend(adj_rgb)

        objects.append(obj_id)
        obj_id += 1

    # Initialize Cutie tracking with first frame + label_mask
    first_t = to_tensor(Image.fromarray(first)).to(DEVICE).float()
    mask_t  = torch.from_numpy(label_mask).to(DEVICE)
    processor.step(first_t, mask_t, objects=objects)

    # Video writer
    fps    = 10  # can be modified as needed
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

    # Frame-by-frame tracking + overlay
    for frame in frames:
        img_t = to_tensor(Image.fromarray(frame)).to(DEVICE).float()
        with torch.no_grad():
            prob = processor.step(img_t)
        m = processor.output_prob_to_mask(prob)
        if m.ndim == 3: m = m[0]
        label = m.cpu().numpy().astype(np.uint8)

        # Generate colored mask, then convert to BGR
        mask_pil = Image.fromarray(label)
        mask_pil.putpalette(palette)
        mask_rgb = np.array(mask_pil.convert("RGB"), dtype=np.uint8)
        mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)

        # Fixed transparency overlay
        overlay = frame.astype(np.float32)
        nz = label > 0
        overlay[nz] = overlay[nz] * (1 - ALPHA) + mask_bgr[nz] * ALPHA

        writer.write(overlay.astype(np.uint8))

    writer.release()
    return out_path


def load_box_records(jsonl_path: Path) -> list[dict]:
    """Read records from info.jsonl"""
    return [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]

def save_box_records(records: list[dict], jsonl_path: Path):
    """Save modified records to new JSONL"""
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main():
    sam_model = SAM(SAM_WEIGHTS)
    for src_sub in sorted(SRC_ROOT.iterdir()):
        if not src_sub.is_dir():
            continue
        dst_sub = DEST_ROOT / src_sub.name
        dst_sub.mkdir(parents=True, exist_ok=True)


        records = load_box_records(src_sub/"info.jsonl")
        
        info_in  = src_sub / "info.jsonl"
        records  = load_box_records(info_in)
        records_to_process = [r for r in records if not r.get("new_video")]
        for rec in records_to_process:
            GlobalHydra.instance().clear()
            cutie_net = get_default_model().to(DEVICE).eval()
            processor = InferenceCore(cutie_net, cfg=cutie_net.cfg)
            processor.max_internal_size = MAX_INTERNAL
            # Construct box_data same as before
            bd = {"objs":{}, "loca":{}}
            for b in rec["box"]:
                cat = "objs" if b["color"]=="blue" else "loca"
                bd[cat].setdefault(rec["video"], []).append(b["bbox"])
            # Construct palette for this video
            palette = [0,0,0]
            for b in rec["box"]:
                palette.extend(COLOR_MAP.get(b["color"], (255,255,255)))
            out_name = f"{Path(rec['video']).stem}_mask.mp4"
            out_vid  = dst_sub / out_name

            process_one_video(
                video_path = src_sub/rec["video"],
                box_data   = bd,
                output_path= out_vid,
                sam_model  = sam_model,
                processor  = processor,
                base_palette = palette
            )
            rec["new_video"] = out_name
            
        save_box_records(records, dst_sub/"info_processed.jsonl")

# import os
# import cv2
# import math
import time
import random
# import numpy as np
# import torch

# Assume your three APIs are already imported in scope:
# - init_models_and_params
# - init_processor_first_step
# - track_frame_and_overlay

def smoke_test_cutie_pipeline(
    video_path: str,
    out_dir: str = "./cutie_smoke_outputs",
    *,
    max_frames: int = 60,       # how many frames to process at most
    dirt_count: int = 3,        # number of random boxes for first-frame init
    seed: int = 0,
    device: str = "cuda:0",
    alpha: float = 0.6,         # overlay alpha
) -> dict:
    """
    Smoke-test the 3 main APIs end-to-end on a given video.
    Pipeline:
      1) open video & read first frame
      2) generate random 'dirt' (N bboxes with color/score) on the first frame
      3) init models (SAM + Cutie InferenceCore)
      4) init_processor_first_step(...) with 'dirt' -> prime Cutie
      5) iterate frames -> track_frame_and_overlay -> save visualized frames

    Returns a small dict of stats for quick sanity check.
    """
    assert os.path.isfile(video_path), f"Video not found: {video_path}"
    os.makedirs(out_dir, exist_ok=True)

    # 1) Read the first frame
    cap = cv2.VideoCapture(video_path)
    ok, first_bgr = cap.read()
    assert ok and first_bgr is not None, "Failed to read the first frame."
    H, W = first_bgr.shape[:2]
    assert H > 0 and W > 0, "Invalid frame size."

    # Cutie uses RGB in its standard demos; we’ll keep first_frame as RGB for your APIs.
    first_rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)

    # 2) Random 'dirt' boxes for initialization
    rng = random.Random(seed)
    def _rand_color():
        return (rng.randint(64, 255), rng.randint(64, 255), rng.randint(64, 255))  # avoid too-dark colors

    def _rand_box():
        # sample center/size in a reasonable range
        cx = rng.uniform(0.2 * W, 0.8 * W)
        cy = rng.uniform(0.2 * H, 0.8 * H)
        bw = rng.uniform(0.1 * W, 0.3 * W)
        bh = rng.uniform(0.1 * H, 0.3 * H)
        x1 = int(max(0, cx - bw / 2)); y1 = int(max(0, cy - bh / 2))
        x2 = int(min(W, cx + bw / 2)); y2 = int(min(H, cy + bh / 2))
        if x2 <= x1: x2 = min(W, x1 + 5)
        if y2 <= y1: y2 = min(H, y1 + 5)
        return [x1, y1, x2, y2]

    dirt = []
    for _ in range(max(1, dirt_count)):
        bbox = _rand_box()
        entry = {
            "box": bbox,                 # [x1,y1,x2,y2]
            "score": float(rng.uniform(0.6, 0.95)),
            "color": _rand_color(),      # (r,g,b)
        }
        dirt.append(entry)

    # 3) Init models (SAM + Cutie InferenceCore)
    t0 = time.time()
    sam_model, processor, device_used = init_models_and_params(device=device)
    assert device_used == device, f"Device mismatch: {device_used} vs {device}"
    t1 = time.time()

    # 4) First-step init (SAM to mask -> Cutie prime)
    Cutie_processor, palette, objects = init_processor_first_step(
        sam_model=sam_model,
        processor=processor,
        first_frame=first_rgb,   # RGB numpy HxWx3
        dirt=dirt,
        device=device_used,
    )
    assert isinstance(objects, list) and all(isinstance(o, int) for o in objects), "Invalid 'objects' returned."
    assert isinstance(palette, list) and len(palette) == 768, "Palette must be a flat list length 768."
    t2 = time.time()

    # 5) Iterate and track
    saved = 0
    total = 1  # we already read 1 frame
    frame_idx = 0
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out_vid_path = os.path.join(out_dir, "overlay.mp4")
    writer = None

    # Save the first overlay frame as sanity (reuse track for consistent path)
    overlay0 = track_frame_and_overlay(first_bgr.copy(), Cutie_processor, palette, device=device_used, alpha=alpha)
    cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx:06d}.jpg"), overlay0)
    saved += 1
    frame_idx += 1

    while total < max_frames:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break
        total += 1

        # Track & overlay (processor.step(image) is called inside your segment_and_track)
        overlay = track_frame_and_overlay(bgr, Cutie_processor, palette, device=device_used, alpha=alpha)

        # Lazy init video writer once we know size
        if writer is None:
            H2, W2 = overlay.shape[:2]
            writer = cv2.VideoWriter(out_vid_path, fourcc, 20.0, (W2, H2))
        writer.write(overlay)

        # Also save a few JPEGs every ~10 frames
        if frame_idx % 10 == 0:
            cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx:06d}.jpg"), overlay)
            saved += 1

        frame_idx += 1

    if writer is not None:
        writer.release()
    cap.release()

    t3 = time.time()
    stats = {
        "video_path": video_path,
        "out_dir": out_dir,
        "objects": objects,
        "num_saved_images": saved,
        "overlay_video": out_vid_path if os.path.isfile(out_vid_path) else None,
        "t_load_models_sec": round(t1 - t0, 3),
        "t_first_step_sec": round(t2 - t1, 3),
        "t_tracking_sec": round(t3 - t2, 3),
        "num_frames_processed": total,
        "frame_size": (H, W),
    }
    print("[SmokeTest] Stats:", stats)
    return stats


if __name__ == "__main__":
    stats = smoke_test_cutie_pipeline(
    video_path="path/to/your/LIBERO/datasets-copy/00_all/libero_10_no_noops/10_00008-of-00032.mp4",
    out_dir="path/to/your/script/test/testvideo/cutie_test_out",
    max_frames=80,
    dirt_count=3,
    seed=42,
    device="cuda:0",
    alpha=0.6,
    )

    # main()