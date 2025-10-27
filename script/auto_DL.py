import cv2
import os
from matplotlib import text
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Glm4vForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import os
import glob
import numpy as np
import shutil
import subprocess

global qwenpath, KB_JSON, output_path, exp_dir

KB_JSON = "path/to/your/simple_exp/KNOW.json"  # knowledge base JSON
output_path = "path/to/your/simple_exp/combine"
exp_dir = "path/to/your/simple_exp"  # example images storage path
qwenpath = "path/to/your/Qwen2.5-VL/model"
qwenpath7B = "path/to/your/Qwen2.5-VL/7B-model"


def load_and_resize(path: str, max_side: int = 512):
    """Load image, resize keeping aspect ratio so max(w,h)<=max_side, return resized img and scale."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    
    if w <= max_side and h <= max_side:
        return img, 1.0, (w, h)  # No resize needed
    
    scale = min(max_side / w, max_side / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    resized.save(path)
    return resized, scale, (w, h)

def combine_images_to_grid(image_folder):
    """
    Combine six images from a folder into a single 2x3 grid image.
    Each image is padded to a square and then resized to 512x512 before placement.

    :param image_folder: Path to the folder containing image files.
    :returns: Path to the saved combined image (uses global `output_path`).
    :raises ValueError: If the folder contains fewer than 6 images.
    """
    
    max_saved=25
        # before save, if the output folder has more than max_saved images, delete the oldest
    out_dir = os.path.dirname(output_path) or "."
    # Match common image formats
    existing = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp"):
        existing.extend(glob.glob(os.path.join(out_dir, ext)))
    # Sort by modification time
    existing.sort(key=lambda p: os.path.getmtime(p))
    # delete the oldest,till max_saved
    while len(existing) >= max_saved:
        oldest = existing.pop(0)
        try:
            os.remove(oldest)
        except OSError:
            pass  # pass
    
    # get all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    image_files = image_files[:6]  # Ensure at most 6 images are taken

    if len(image_files) < 6:
        raise ValueError("Not enough images found in the folder")

    rows = 2
    cols = 3
    tile_size = 512
    grid_image = Image.new('RGB', (cols * tile_size, rows * tile_size), color=(0, 0, 0))

    for i, image_file in enumerate(image_files):
        if i >= rows * cols:
            break  # Only arrange the first 6 images
        image_path = os.path.join(image_folder, image_file)
        with Image.open(image_path) as img:
            # Pad the image to a square
            width, height = img.size
            max_dim = max(width, height)
            padded_img = Image.new('RGB', (max_dim, max_dim), color=(0, 0, 0))
            x_offset = (max_dim - width) // 2
            y_offset = (max_dim - height) // 2
            padded_img.paste(img, (x_offset, y_offset))

            # Resize to 512x512
            tile = padded_img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)

            # Calculate paste position in the grid
            col_idx = i % cols
            row_idx = i // cols
            x = col_idx * tile_size
            y = row_idx * tile_size

            grid_image.paste(tile, (x, y))

    # Get current time and format as y-m
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save the combined image with the current timestamp
    final_output_path = os.path.join(output_path, f"{timestamp}.png")
    grid_image.save(final_output_path)
    print(f"Combined image saved to {final_output_path}")
    return final_output_path

def download_images(currentimage,query,processor_GLM,model_GLM):
    """
    download images based on keywords to a specified directory.

    :param keywords: keywords to search for
    :param dir_photo: directory to save the images
    """
    
    system_prompt = rf"""
    You are an intelligent assistant specialized in analyzing images and extracting meaningful information. Your task is to identify a specific person or object that appears in all provided images and generate five of the most relevant keywords to describe this person or object.
    **Think in ten sentences.** You must follow this rule strictly.
    Guidelines:
    For the combined image:
    If the same person appears in all images:
    Focus on describing the person's gender, skin tone, and occupation.
    Avoid keywords related to clothing or environment.
    Example keywords might include: "female", "light-skinned", "doctor", etc.
    If the same object appears in all images:
    Focus on describing the object's physical characteristics.
    Example keywords might include: "round", "metallic", "small", etc.
    **IMPORTANT** the keywords are going to help another Model to find the same or almost like subjects or person in the real world image,
    thus the keywords should be very specific and descriptive, not general or abstract, can reflect the basic attributes of this task or thing.
    Making another VLM easily find the same or similar subjects or person in the real world image.

    For the current image:
    There are something suit for the query"{query}" but the model cannt find bbox exactly.
    Your mission is to base on the current image and combined image to descrebe the most same thing in them.
    
    Output Format:
    Output the keywords in JSON format.
    Ensure the output contains only the keywords, without additional text or explanation.
    The JSON structure should be a list of strings.
    Example JSON Output:["female", "light-skinned", "doctor", "middle-aged", "smiling"].
    Your output should in a format that the code below can easily extra the keywords:
    --match = re.search(r"\[.*?\]", output_text[0])
    --  if match:
    --      str_list = json.loads(match.group(0))
    --      print(str_list)

    Task:
    Analyze the provided images and generate five keywords that best describe the identified person or object based on the guidelines above. 
    Output the keywords in the specified JSON format.
    input:{query}
    output:
    """
    dir_photo = query
    download_dir = os.path.join(exp_dir, dir_photo) #experience photo storage address
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    else:
        # If the directory already exists, clear all files in the directory
        for file in os.listdir(download_dir):
            file_path = os.path.join(download_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    download_script = f"""bbid.py "{query}" -o "{download_dir}" --limit 8"""
    clean_env = dict(os.environ)
    for k in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'all_proxy'):
        clean_env.pop(k, None)
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            if attempt < 2:
                subprocess.run(download_script, shell=True, check=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            )
                break
            else:
                subprocess.run(download_script, shell=True, check=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            env=clean_env)
                break
        except subprocess.CalledProcessError as e:
            print(f"[{attempt+1}/{max_attempts}] download failure: {e.stderr.decode().strip()}")
            if attempt == max_attempts - 1:
                shutil.rmtree(download_dir, ignore_errors=True)
                return "Download failed", "Failed"
    # Get all image files in the folder
    print("Download Successful.")
    image_path=combine_images_to_grid(download_dir)
    load_and_resize(image_path, max_side=1280)
    h,w = Image.open(image_path).size
    com_image = Image.open(image_path).resize((h,w), Image.Resampling.LANCZOS)
    cur_image = Image.fromarray(currentimage).resize(size=(h,w))
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type":"text",
                    "text":"Here is the combined image from web.",
                },
                {
                    "type": "image",
                    "image": com_image,  ##here input a combined photo
                },
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type":"text",
                    "text":"This is the current image from camera.",
                },
                {
                    "type": "image",
                    "image": cur_image,  ##here input a current photo
                },
            ]
        }
    ]

        
    output_text = useglmt(messages,processor_GLM,model_GLM)
    import re
    try:
        match = re.search(r"\[.*?\]", output_text[0])
    except:
        None
    if match:
        str_list = json.loads(match.group(0))
        print(str_list)
    else:
        try:
            m = re.search(r'<answer>(.*?)</answer>', output_text, flags=re.S)
            if m:
                str_list = json.loads(m.group(1))   # 2. Python list
                print(str_list)
        except:
            return "Failed","Failed"
    if str_list:
        mainwords = str_list
    else:
        return "Failed","Failed"
    append_knowledge(
        filepath=KB_JSON,
        knowledge_name=query,
        keywords=mainwords,
        folder_path=download_dir
        )
    return mainwords,download_dir
    
import json
from pathlib import Path
from typing import List, Union, Dict, Any

import json
import random
from pathlib import Path
from typing import List, Dict, Any

# -------------------------------- 1. append, renew --------------------------------
def append_knowledge(
    filepath: Path,
    knowledge_name: str,
    keywords: List[str],
    folder_path: Path,
) -> None:
    """
    Append or update a record in the knowledge base JSON:
      {
        "knowledge_name": {
            "keywords": [...3 keywords...],
            "folder":   "/abs/path/to/img_dir"
        }
      }
    """
    filepath = Path(filepath)
    data: Dict[str, Any] = {}
    if filepath.exists():
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass  # ignore

    # Write / Update
    data[knowledge_name] = {
        "keywords": list(dict.fromkeys(keywords))[:5],  # Remove duplicates and take the first 5
        "folder": folder_path,
    }

    filepath.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ------------------------------ 2. Query and return 6 images ------------------------------
def get_images_for_knowledge(
    filepath: Path,
    current_image:Union[np.ndarray,Image.Image],
    query: str,  # existing download function
    processor_GLM,model_GLM
) -> List[str]:
    """
    Return 6 image paths based on `query` (can be knowledge name or keywords).
    - Knowledge name hit: directly take 6 images randomly/evenly from that directory
    - Keyword hit: evenly sample from all matched directories to get 6 images
    - No match: call downloadKL(query) -> return (knowledge_name, keywords, folder)
                then write to knowledge base and sample 6 from new directory
    """
    filepath = Path(filepath)
    data: Dict[str, Any] = {}
    if filepath.exists():
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    if isinstance(current_image, Image.Image):
        current_image = cv2.cvtColor(np.array(current_image), cv2.COLOR_RGB2BGR)
    # def pick_n(img_list: List[Path], n: int) -> List[str]:
    #     if len(img_list) <= n:
    #         return [str(p) for p in img_list]
    #     step = len(img_list) / n
    #     return [str(img_list[int(i * step)]) for i in range(n)]

    # ---------- 2.1 Direct knowledge name hit ----------
    if query in data:
        return data[query].get("keywords", []), data[query].get("folder", [])

    # # ---------- 2.2 Keyword reverse lookup ----------
    # hit_dirs = []
    # for name, info in data.items():
    #     if query in info.get("keywords", []):
    #         hit_dirs.append(Path(info["folder"]))

    # if hit_dirs:
    #     # Even sampling
    #     per_dir = max(1, 6 // len(hit_dirs))
    #     result = []
    #     for d in hit_dirs:
    #         imgs = sorted([p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
    #         result.extend(pick_n(imgs, per_dir))
    #     # If less than 6, randomly fill up
    #     if len(result) < 6:
    #         extra_pool = []
    #         for d in hit_dirs:
    #             extra_pool.extend(
    #                 [p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}]
    #             )
    #         random.shuffle(extra_pool)
    #         for p in extra_pool:
    #             if len(result) >= 6:
    #                 break
    #             if str(p) not in result:
    #                 result.append(str(p))
    #     return result[:6]

    # ---------- 2.3 Complete miss → call downloadKL ----------
    # Assume downloadKL returns (new_name, new_keywords, new_folder)
    try:
        new_keywords, new_folder = download_images(current_image,query,processor_GLM,model_GLM) # Need to modify this
    except Exception as e:
        print(f"Download failed: {e}")
        return "Failed", "Failed"
    if new_folder == "Failed" or new_keywords == "Failed":
        return [query], "no result"  # Prevent download failure due to poor network connection
    else:
        return new_keywords,new_folder

def useqwen(messages):
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwenpath,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    min_pixels = 64*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(qwenpath, min_pixels=min_pixels, max_pixels=max_pixels,use_fast=True)
    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1000)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

def search_reflect(current_image:Union[np.ndarray,Image.Image] , query:str,processor_GLM,model_GLM):
    pics,folder = get_images_for_knowledge(KB_JSON,current_image, query,processor_GLM,model_GLM)
    if pics == "Failed":
        print("Download failed, please check network connection or keywords.")
        return "Failed", "Failed"
    if folder == "no result":
        return pics, None
    comimage = combine_images_to_grid(folder) # PIL format image
    print("Returned 5 keywords")
    print(pics)
    print("Returned image folder")
    print(folder)
    print("Composite image path")
    print(comimage)
    return pics,comimage # keywords and composite image path

### DINO decoder:
def keep_larger_on_overlap(boxes, scores, labels, overlap_thresh=0.5):
    """
    Apply "overlap area" suppression to a group of candidate boxes:
    If the intersection area of two boxes exceeds overlap_thresh of the smaller box area,
    discard the smaller box; otherwise keep both.

    Parameters:
    - boxes: List[List[int]]  Each [x0,y0,x1,y1]
    - scores: List[float]
    - labels: List[str]
    - overlap_thresh: float  (0~1)

    Returns:
    - keep_idxs: List[int]  Indices of finally kept boxes
    """
    N = len(boxes)
    keep = [True] * N

    # Calculate area of each box
    areas = [(x1 - x0) * (y1 - y0) for (x0,y0,x1,y1) in boxes]

    # Pairwise comparison
    for i in range(N):
        if not keep[i]:
            continue
        for j in range(i+1, N):
            if not keep[j]:
                continue

            # Calculate intersection
            x0 = max(boxes[i][0], boxes[j][0])
            y0 = max(boxes[i][1], boxes[j][1])
            x1 = min(boxes[i][2], boxes[j][2])
            y1 = min(boxes[i][3], boxes[j][3])
            w = max(0, x1 - x0)
            h = max(0, y1 - y0)
            inter = w * h
            if inter == 0:
                continue

            # Smaller box area
            min_area = min(areas[i], areas[j])
            if inter > overlap_thresh * min_area:
                # Discard the one with smaller area
                if areas[i] > areas[j]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break  # i is discarded, break inner loop

    return [idx for idx, v in enumerate(keep) if v]



def detect_and_draw_smart(image_path, prompt,
                          box_thresh=0.3, text_thresh=0.3,
                          overlap_thresh=0.5):
    """
    Use Grounding DINO detection + custom "keep larger box" suppression, then draw.
    """
    
    import numpy as np
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    MODEL_NAME = "path/to/your/smolVLM/model/grounddinov1"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_NAME).to(DEVICE)
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
    elif isinstance(image_path, np.ndarray):
        img = image_path
    elif isinstance(image_path, Image.Image):  # Assuming PIL is imported as Image
        img = cv2.cvtColor(np.array(image_path), cv2.COLOR_RGB2BGR)
    else:
        raise TypeError("Unsupported image type. Provide a file path, numpy array, or PIL image.")

    if img is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    H, W = img.shape[:2]

    # 1. Inference
    inputs = processor(images=img, text=[prompt], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)

    # 2. Post-processing
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        box_threshold=box_thresh,
        text_threshold=text_thresh,
        target_sizes=[(H, W)]
    )

    if (
        not results or
        not isinstance(results, (list, tuple)) or
        len(results) == 0 or
        not isinstance(results[0], dict) or
        "boxes" not in results[0] or
        not isinstance(results[0]["boxes"], torch.Tensor) or
        results[0]["boxes"].numel() == 0
        ): # Fallback when nothing is found
        return []
    
    # 3. Collect candidate boxes
    all_boxes, all_scores, all_labels = [], [], []
    for res in results:
        boxes  = res["boxes"].cpu().numpy().astype(int)
        scores = res["scores"].cpu().numpy()
        labels = res["text_labels"]
        for (x0,y0,x1,y1), s, lbl in zip(boxes, scores, labels):
            if s >= box_thresh:
                all_boxes.append([x0,y0,x1,y1])
                all_scores.append(float(s))
                all_labels.append(lbl)

    # 4. Custom suppression: keep larger boxes
    keep_idxs = keep_larger_on_overlap(all_boxes, all_scores, all_labels, overlap_thresh)

    # 5. Draw
    result_img = img.copy()
    for i in keep_idxs:
        x0,y0,x1,y1 = all_boxes[i]
        s   = all_scores[i]
        lbl = all_labels[i]
        color = tuple(np.random.randint(0,256,3).tolist())

        cv2.rectangle(result_img, (x0,y0), (x1,y1), color, 2)
        text = f"{lbl}:{s:.2f}"
        (w,h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(result_img,
                      (x0, y0 - h - 4),
                      (x0 + w + 4, y0),
                      color, -1)
        cv2.putText(result_img, text,
                    (x0 + 2, y0 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255,255,255), 1,
                    lineType=cv2.LINE_AA)

    result = {
        "result_image": result_img,
        "original_image": img,
        "boxes_info": [
            {
                "box": [int(x) for x in all_boxes[i]],
                "score": float(all_scores[i]),
                "label": all_labels[i]
            }
            for i in keep_idxs
        ]
    }
    return result

def dino_decoder(image:any,textprompt: str):
    out = detect_and_draw_smart(
        image, textprompt,
        box_thresh=0.3, text_thresh=0.3
    )
    return out

def initGLMT():
    model_path = "path/to/your/smolVLM/model/GLMThinking/ZhipuAI/GLM-4.1V-9B-Thinking"

    """
    This model has quite high VRAM requirements
    """

    MODEL_PATH = model_path
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
    model = Glm4vForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to("cuda:0")
    print("Finish init GLM")
    return processor,model

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


# ------------------------------ Example usage ------------------------------
if __name__ == "__main__":

    # # Assume downloadKL(query) implementation exists
    # def downloadKL(query: str):
    #     """
    #     Example downloadKL, only demonstrates return format.
    #     Should actually download images to new_folder and return its path.
    #     """
    #     new_folder = Path(f"./imgs/{query}")
    #     new_folder.mkdir(parents=True, exist_ok=True)
    #     # … download images to new_folder …
    #     return query, [f"{query}_kw1", f"{query}_kw2", f"{query}_kw3"], new_folder

    # Read test
    import cv2
    cur_image = cv2.imread("path/to/your/LIBERO/test/testimage/org_image/obj_00423-of-00032.jpg")
    GLM_pro, GLM_model = initGLMT()  # Initialize GLM model
    pics,folder = get_images_for_knowledge(KB_JSON,cur_image, "panda",GLM_pro,GLM_model)
    if pics == "Failed":
        print("Download failed, please check network connection")
        exit(1)
    if folder == "no result":
        print("Thinking time too long")
    else:
        comimage = combine_images_to_grid(folder)
        print("Returned 5 keywords")
        print(pics)
        print("Returned image folder")
        print(folder)
        print("Composite image path")
        print(comimage)

    
