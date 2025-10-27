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
# Replace these placeholders with your repository-relative or absolute paths.
SRC_ROOT     = Path("path/to/your/boxdatas")    # Source data root
DEST_ROOT    = Path("path/to/your/seg-video")   # Output root
SAM_WEIGHTS  = "path/to/your/sam2.1_l.pt"       # SAM weights file
DEVICE       = "cuda:0"
MAX_INTERNAL = -1
ALPHA        = 0.35                               # Overlay alpha for masks
# Strength used when adjusting color brightness
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
    PIL's putpalette requires a flat [r,g,b,...] list of length 768.
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
      sam_model: Ultralytics SAM model (already to(device).eval())
      processor: Cutie InferenceCore (not yet initialized with the first frame)
      device:    Current device string

        Function 1: Initialize all models and parameters (without performing the first-frame step).
        Returns:
            sam_model: Ultralytics SAM model (moved to device and set to eval)
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

    Function 2: Initialize GlobalHydra and complete Cutie's first step using "first frame + mask + object IDs".
    - You can directly provide label_mask (H,W uint8) and objects (ID list), or
    - Provide dirt list (containing box/score/color), or
    - Provide box_data structure ({"objs":{video:[bboxes...]}, "loca":{...}})
    This function will:
      1) GlobalHydra.instance().clear()
      2) If needed, call SAM to generate label_mask based on first frame and boxes
      3) Assemble/pad palette
      4) Call processor.step(first_t, mask_t, objects=objects)
    Returns:
      processor (completed first frame initialization)
      palette (PIL flat palette, including background black)
      objects (object ID list, corresponding to values in mask)
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

                        # 分配颜色：优先用 COLOR_MAP，否则随机
                        if cls == "objs":
                            base_rgb = COLOR_MAP.get("blue", (255, 0, 0))
                        else:
                            base_rgb = COLOR_MAP.get("red", (0, 0, 255))
                        palette.extend(list(base_rgb))
                        objects.append(next_id)
                        next_id += 1
            built_from_boxes = True

        else:

            pass

        if len(objects) == 0 and built_from_boxes:
            # 防御：有框却没写入成功
            uniq = np.unique(label_mask)
            objects = [int(v) for v in uniq if v != 0]

    if objects is None or len(objects) == 0:
        uniq = np.unique(label_mask)
        objects = [int(v) for v in uniq if v != 0]


    palette = _ensure_pil_palette(palette)

    # 3) Cutie first frame step
    first_t = to_tensor(Image.fromarray(first_frame)).to(device).float()
    mask_t = torch.from_numpy(label_mask).to(device)
    processor.step(first_t, mask_t, objects=objects)

    return processor, palette, objects

def init_processor_first_step_v2(
    sam_model: "SAM",
    processor: "InferenceCore",
    first_frame: np.ndarray,
    *,
    # one of label_mask+objects；or input dirt；or provide box_data
    label_mask: np.ndarray | None = None,
    objects: list[int] | None = None,
    palette: list[int] | None = None,
    dirt: list[dict] | None = None,
    box_data: dict | None = None,
    device: str = DEVICE,
) -> tuple["InferenceCore", list[int], list[int]]:
    """
    Improvements:
      - Only assign next_id / append to the palette / append to objects when the SAM mask
        for the bbox actually wrote pixels (write_idx.any() == True).
      - Removed color adjustment by score; strictly use entry["color"] (or the fixed COLOR_MAP color),
        ensuring the RGB matches the declared color name.
    """
    # 1) reset Hydra
    GlobalHydra.instance().clear()
    cutie_net = get_default_model().to(device).eval()
    processor = InferenceCore(cutie_net, cfg=cutie_net.cfg)
    H, W = first_frame.shape[:2]

    # 2) generate label_mask / objects / palette
    if palette is None:
        palette = [0, 0, 0]  # background black

    built_from_boxes = False

    if label_mask is None:
        label_mask = np.zeros((H, W), dtype=np.uint8)
        objects = [] if objects is None else objects
        next_id = 1 if len(objects) == 0 else (max(objects) + 1)

        if dirt is not None:
            # dirt: [{"box":[x1,y1,x2,y2] or [-1...], "score":float, "color":(r,g,b), "color_name": str}, ...]
            for entry in dirt:
                bbox = entry.get("box", [])
                if bbox == [-1, -1, -1, -1] or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = map(int, bbox)

                res = sam_model.predict(source=first_frame, device=device, bboxes=[[x1, y1, x2, y2]])
                if not res or res[0].masks is None:
                    continue
                mask_full = (res[0].masks.data[0].cpu().numpy() > 0.5)

                # restrict ROI and try to write
                y1c, y2c = max(0, y1), min(H, y2)
                x1c, x2c = max(0, x1), min(W, x2)
                if y2c <= y1c or x2c <= x1c:
                    continue

                roi = mask_full[y1c:y2c, x1c:x2c]
                region = label_mask[y1c:y2c, x1c:x2c]
                write_idx = roi & (region == 0)

                # ✅ Only assign ID / color when pixels are actually written
                if write_idx.any():
                    region[write_idx] = next_id
                    label_mask[y1c:y2c, x1c:x2c] = region

                    base_rgb = tuple(entry.get("color", (255, 255, 255)))
                    palette.extend(list(base_rgb))
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

                        y1c, y2c = max(0, y1), min(H, y2)
                        x1c, x2c = max(0, x1), min(W, x2)
                        if y2c <= y1c or x2c <= x1c:
                            continue

                        roi = mask_full[y1c:y2c, x1c:x2c]
                        region = label_mask[y1c:y2c, x1c:x2c]
                        write_idx = roi & (region == 0)

                        if write_idx.any():
                            region[write_idx] = next_id
                            label_mask[y1c:y2c, x1c:x2c] = region

                            # Assign fixed colors: maintain the old logic of "objs blue / loca red"
                            if cls == "objs":
                                base_rgb = COLOR_MAP.get("blue", (0, 0, 255))
                            else:
                                base_rgb = COLOR_MAP.get("red", (255, 0, 0))
                            palette.extend(list(base_rgb))
                            objects.append(next_id)
                            next_id += 1

            built_from_boxes = True

        else:
            # No box information provided, and no mask given; keep empty mask and empty objects (generally not recommended)
            pass

        # If boxes were indeed constructed but objects are still empty, fall back on mask (extreme fallback)
        if len(objects) == 0 and built_from_boxes:
            uniq = np.unique(label_mask)
            objects = [int(v) for v in uniq if v != 0]

    # If the user directly provides label_mask but not objects, infer from mask
    if objects is None or len(objects) == 0:
        uniq = np.unique(label_mask)
        objects = [int(v) for v in uniq if v != 0]

    # palette compensates for the format required by PIL
    palette = _ensure_pil_palette(palette)

    # —— Optional consistency self-check (enable during development, comment out after stabilization) ——
    # Ensure the number of objects matches the number of color entries in the palette (excluding background)
    try:
        assert len(objects) == (len(palette) // 3 - 1)
        mask_ids = set(np.unique(label_mask)) - {0}
        assert mask_ids.issubset(set(objects))
    except AssertionError:
        print("[init_v2 WARN] objects / palette / mask IDs mismatch - please inspect upstream.")

    # 3) Cutie first frame step
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

    function 3: Input current frame, output image with overlaid tracking masks (BGR).
    reuse segment_and_track, but ensure the palette is safely padded first.
    """
    # segment_and_track uses global ALPHA; this allows temporary external overrides
    global ALPHA
    old_alpha = ALPHA
    ALPHA = alpha
    try:
        pal = _ensure_pil_palette(palette)
        out = segment_and_track(frame, processor, pal)  # return uint8 BGR
    finally:
        ALPHA = old_alpha
    return out
# ========== end of three main functions ==========


def reset_processor_memory(processor: InferenceCore):
    """Clear Cutie's internal memory features"""
    processor.memory.clear_sensory_memory()      # Clear image_feature_store
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
    mask = mask.detach().clone()   # ← Force detach from computation graph
    label = mask.cpu().numpy().astype(np.uint8)

    mask_pil = Image.fromarray(label)
    mask_pil.putpalette(palette)
    mask_color = np.array(mask_pil.convert("RGB"), dtype=np.uint8)

    out = frame.astype(np.float32)
    nz  = label > 0
    out[nz] = out[nz] * (1-ALPHA) + mask_color[nz] * ALPHA
    return out.astype(np.uint8)

def adjust_color_by_score(rgb_color: tuple[int,int,int], score: float) -> tuple[int,int,int]:
    """
    according to score [0,1] adjust rgb_color brightness:
      - score > 0.5 → darker color
      - score < 0.5 → lighter color
    Use HSV color space for linear mapping of V channel.
    """
    # 1. Normalize to [0,1]
    r, g, b = [c / 255.0 for c in rgb_color]
    # 2. Convert to HSV
    hsv = cv2.cvtColor(
        np.array([[[r, g, b]]], dtype=np.float32),
        cv2.COLOR_RGB2HSV
    )[0,0]
    h, s, v = hsv
    # 3. Calculate brightness factor: score=1→v*(1-0.5*strength)；score=0→v*(1+0.5*strength)
    v_factor = 1.0 - (score - 0.5) * HSV_STRENGTH * 2
    v_factor = float(np.clip(v_factor, 0.0, 2.0))
    # 4. Apply and convert back to RGB
    hsv_adj = np.array([[[h, s, v * v_factor]]], dtype=np.float32)
    rgb_adj = cv2.cvtColor(hsv_adj, cv2.COLOR_HSV2RGB)[0,0]
    return tuple((rgb_adj * 255).astype(np.uint8))


if __name__ == "__main__":
    import cv2
    from ultralytics import SAM