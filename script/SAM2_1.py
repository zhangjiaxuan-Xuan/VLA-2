import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from ultralytics import SAM
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="cutie")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp")

# ---------- Global constants ----------
# Replace these placeholders with your actual paths if needed
SAM2_WEIGHTS = "path/to/your/sam2.1_b.pt"  # replace with your weights
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA        = 0.55                         # overlay opacity

@torch.inference_mode()
def build_processor_and_palette(frame0_bgr: np.ndarray,
                                roi_color_dict: dict):
    """
    First frame initialization: SAM2 generates masks + Cutie initialization
    :param frame0_bgr: first frame BGR
    :param roi_color_dict: {"name": {"box":[x1,y1,x2,y2], "color":(B,G,R)}, ...}
    :return: (processor, palette) for reuse in subsequent frames
    """
    # 1. SAM2
    sam2 = SAM(SAM2_WEIGHTS)
    H, W = frame0_bgr.shape[:2]

    label_mask = np.zeros((H, W), dtype=np.uint8)
    palette = [0, 0, 0]  # background black
    objects, obj_id = [], 1

    for _, spec in roi_color_dict.items():
        x1, y1, x2, y2 = spec["box"]
        color_bgr = spec["color"]

        # SAM2 generates segmentation
        sam_res = sam2.predict(source=frame0_bgr, device=DEVICE,
                               bboxes=[[x1, y1, x2, y2]])
        mask_full = sam_res[0].masks.data[0].cpu().numpy() > 0.5

        # Write label_mask
        label_mask[(mask_full) & (label_mask == 0)] = obj_id
        b, g, r = color_bgr
        palette.extend([r, g, b])  # palette requires RGB order
        objects.append(obj_id)
        obj_id += 1

    # 2. Cutie initialization
    cutie = get_default_model()
    processor = InferenceCore(cutie, cfg=cutie.cfg)
    first_tensor = to_tensor(Image.fromarray(frame0_bgr)).to(DEVICE).float()
    label_tensor = torch.from_numpy(label_mask).to(DEVICE)
    processor.step(first_tensor, label_tensor, objects=objects)
    return processor, palette

@torch.inference_mode()
def segment_and_track_one_frame(frame_bgr: np.ndarray,
                                processor: InferenceCore,
                                palette: list) -> np.ndarray:
    """
    Single frame inference: return overlay (BGR)
    """
    img_tensor = to_tensor(Image.fromarray(frame_bgr)).to(DEVICE).float()
    output_prob = processor.step(img_tensor)
    label_mask = processor.output_prob_to_mask(output_prob).cpu().numpy().astype(np.uint8)

    # Colored mask
    mask_pil = Image.fromarray(label_mask)
    mask_pil.putpalette(palette)
    mask_color = np.array(mask_pil.convert("RGB"))

    # overlay
    overlay = frame_bgr.astype(np.float32)
    nonzero = label_mask > 0
    overlay[nonzero] = overlay[nonzero] * (1 - ALPHA) + mask_color[nonzero] * ALPHA
    return overlay.astype(np.uint8)

import cv2,os


if __name__ == "__main__":
    # 1. Prepare dictionary
    ROI_COLOR_DICT = {
        'tennis ball_box_1': {'box': [46, 230, 125, 310], 'color': (255, 99, 71)},
        'tennis ball_box_2': {'box': [151, 220, 219, 294], 'color': (60, 179, 113)},
        'labubu':            {'box': [270,  58, 412, 291], 'color': (30, 144, 255)},
    }

    # Replace src_mp4 with your video path or a repo-relative placeholder
    src_mp4 = "path/to/your/LIBERO/test/video/libero_demo.mp4"
    dst_mp4 = os.path.splitext(src_mp4)[0] + "_overlay.mp4"

    # ---------- 2. Read first frame ----------
    cap = cv2.VideoCapture(src_mp4)
    if not cap.isOpened():
        raise FileNotFoundError(src_mp4)

    ret, frame0 = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame")

    processor, palette = build_processor_and_palette(frame0, ROI_COLOR_DICT)

    # ---------- 3. Prepare video writing ----------
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')   # VS Code can play directly
    writer = cv2.VideoWriter(dst_mp4, fourcc, fps, (w, h))

    # ---------- 4. Process frame by frame ----------
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        overlay = segment_and_track_one_frame(frame, processor, palette)
        writer.write(overlay)          # write frame

    # ---------- 5. Cleanup ----------
    cap.release()
    writer.release()
    print(f"Saved: {dst_mp4}")
    
    ##Modification idea: Initialize outside the function, differentiate between first time receiving box and later using memory segmentation inside the function