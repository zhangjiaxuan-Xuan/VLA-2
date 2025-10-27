from auto_DL import useqwen,search_reflect,dino_decoder
import json
import os, re
from PIL import Image

"""
Knowledge base: path/to/your/simple_exp/KNOW.json
Example images storage: path/to/your/simple_exp
Test model locations (placeholders):
    7B model: path/to/your/Qwen2.5-VL/7B-model
    3B model: path/to/your/Qwen2.5-VL/model

Main entry: from_sub_to_box(mainsubjects, image=None)

Overview:
    For each input main subject, first search the KB. If found, use stored data; otherwise
    attempt to download/search and build a local KB entry. Then use VLM (qwen) to obtain boxes.

Inputs: mainsubjects
Outputs: boxes dict
Reset condition: newsubjects
"""

# Define a color palette for box visualization
objs_box_colors = {
    "subject1": (255, 99, 71),  # Tomato
    "subject2": (60, 179, 113),  # Medium Sea Green
    "subject3": (30, 144, 255),  # Dodger Blue
    "subject4": (255, 215, 0),  # Gold
    "subject5": (255, 105, 180),  # Hot Pink
    # Add more subjects and their corresponding colors as needed
}

loca_box_colors = {
    "subject1": (0, 156, 184),
    "subject2": (195, 76, 142),
    "subject3": (225, 111, 0),
    "subject4": (0, 40, 255),
    "subject5": (0, 150, 75),
}

def extract_bbox(raw_lines):
    """
    raw_lines: list[str]，例如
      ['```json\n{ ... }\n```']
    JSON after read and first bbox_2d
    """
    # 1. take out ```json ... ``` original string
    raw = raw_lines[0]

    # 2. delete ```json beginning ``` lasting
    clean = re.sub(r'^```json\n', '', raw)
    clean = re.sub(r'\n```$', '', clean)
    
    # 3. 解析 JSON
    data = json.loads(clean)
    
    # 4. read bbox_2d（assume at least one box）
    try:
        bbox0 = data['boxes'][0]['bbox_2d']
        answer = data['answer']
        all_bboxes = [ item["bbox_2d"] for item in data["boxes"] ]
    except (IndexError, KeyError, TypeError, json.JSONDecodeError, ValueError, AttributeError):
        bbox0 = data['boxes'][0]
        answer = data['answer']
        all_bboxes = [bbox0]
    
    return answer, data, all_bboxes

def load_and_resize(path: str, max_side: int = 512):
    """Load image, resize keeping aspect ratio so max(w,h)<=max_side, return resized img and scale."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(max_side / w, max_side / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    resized.save(path)  # Save resized image to the same path
    return resized, scale, (w, h)

def from_sub_to_box(mainsubjects:list[str], image:any,obtype="objs"):
    """_summary_

    Args:
        mainsubjects (list[str]): _description_
        image (_type_, optional): _description_. real time image.
        
    input: mainsubjects = ["subject1", "subject2", ...],
           image: PIL.Image or path to image file
    output:boxes= {
        "box1": [x1, y1, x2, y2],
        "box2": [x1, y1, x2, y2],
        ...


    meaning in the sequence of boxes: the first is to be grasped by the gripper, the last is to be placed, the middle ones may be misidentified or unrelated similar objects
    }
    """
    if obtype == "objs":
        box_colors = objs_box_colors
    else:
        box_colors = loca_box_colors
    boxes = {}
    for sub in mainsubjects:
        box = useqwentobox(sub, image)
        if isinstance(box, list) and len(box) > 1: #box：[[1,2,3,4],[4,5,6,7], ...]
            for i, b in enumerate(box):
                boxes[f"{sub}_box_{i+1}"] = b  # Naming multiple boxes with a suffix
        else:
            boxes[sub] = box[0]  # Append the box data for each subject
    #here we need to assign colors to boxes, then filter out -1 boxes
    # Assign colors to boxes
    # Assign colors to boxes
    color_keys = list(box_colors.keys())

    # Assign colors to boxes based on the selected color palette
    for i, sub in enumerate(boxes.keys()):
        if i < len(color_keys):  # Ensure we don't exceed the available colors
            boxes[sub] = {"box": boxes[sub], "color": box_colors[color_keys[i]]}
        else:
            boxes[sub] = {"box": boxes[sub], "color": box_colors[color_keys[-1]]}  # No color available
        # Filter out boxes with -1 and their color labels
    for sub in boxes.keys():
        if boxes[sub]["box"] == [-1, -1, -1, -1]:
            boxes[sub]["box"] = [0, 0, 1, 1]  # Change box to [0, 0, 1, 1] 特殊化，方便后面的VLM特定prompt描述和VLA输入
            boxes[sub]["color"] = (255, 255, 255)  # Change color to white
    ##这里需要将上面的功能函数化并进行测试
    
    return boxes  # Return the boxes dictionary



def useqwentobox(subject, image):
    """Use Qwen to get the box of the subject in the image."""
    outformat = {
        "answer": "~~~",
        "box": [[-1, -1, -1, -1]],
        "center point": [-1, -1],
        "image file name": "~~~"
    }
    outformat2 = {
        "answer": "~~~",
        "box": [[-1, -1, -1, -1],[-1, -1, -1, -1],[-1, -1, -1, -1]],
        "center point": [-1, -1],
        "image file name": "~~~"
    }
    system_prompt = f"""
    You is a smart subjects finder who can find the boxes of the subjects in the image.
    But HONEST is the most important thing, you should not output any wrong information or inaccurate box position.
    Output in your customized json format. If you can not find, output all -1 in boxes[bbox_2d] as [-1,-1,-1,-1].
    Most of time there will be one subject in the image, but there may be multiple subjects in the image.
    Your first output content "answer" should and *ONLY* be "YES" or "NO" to indicate whether you can find the subject in the image. 
    Then are "boxes" and other information.
    For common objects such as "bag","bottle","cup","phone","computer","keyboard","mouse","chair","table","book","pen","pencil","paper", etc., 
    if these common objects are requeried to find in the image, you must find them and answer "YES" with **sound reasoning** .
    """
    ## Output in json format:{outformat} if only one fit. Format in {outformat2} if multiple fit.
    message1 = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (f"Find out the boxes of the subject in the image which mention in the user's text \"{subject}\".",
                            "There maybe multiple subjects in the image fit the discription, you need to box them all depend on satuation.",
                            " If can not find ,output all -1."
                            )
                },
                {
                    "type": "image",
                    "image": image
                }
            ],
        },
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                }
            ]
        }
    ]
    output = useqwen(message1)
    #print(output)
    answer,data,bbox = extract_bbox(output)
    
    if answer=="YES":
        return bbox
    else:
        keywords, comimage = search_reflect(subject) 
        ## the comimage is not resized here
        if keywords == "Failed":
            box = force_findbox(subject, image)
            return box
        else:
            resize_image = load_and_resize(comimage)
            box = findboxaftersearch_dino(subject,keywords, comimage,image)
            return box
    
def findboxaftersearch_dino(subject, keywords, comimage, image):
    textpromt = f"{subject}, {' '.join(keywords)}."
    out = dino_decoder(image, textpromt)
    if out ==[]:
        return [[-1, -1, -1, -1]]
    boxes = [item["box"] for item in out["boxes_info"]]
    return boxes
    
def findboxaftersearch(subject,keywords, resize_image, image):
    """_summary_

    Args:
        subject (_type_): _description_
        resize_image (_type_): _description_
        image (_type_): _description_
        
    imput : subject = "subject name", 
            resize_image = (resized image, scale, original size),
            image: PIL.Image or path to image file
    output: box = [x1, y1, x2, y2]...
    
    """
    outformat = {
        "box": [[-1, -1, -1, -1]   ,...],
        "center point": [-1, -1],
        "image file name": "~~~"
    }
    userprompt = f"""
    The image you recived is the observation of the real world. Please find the box of the {subject} based on keywords {keywords} in the image.
    """
    systemprompt = f"""
    You is a smart subjects finder who can find the boxes of the subjects in the image.
    You **MUST** find the boxes of the subjects in the image based on the keywords and the image.
    The subject you need to find is **{subject}**. the image send to you is a mix of the similar subjects.
    You need to find the subjects based on the most basic elements illustrated by the keywords **{keywords}**, also with the help of the mixed image.
    Output in json format, first output "answer" is defult to "YES", then is boxes[bbox_2d] and other information. 
    multiple boxes is OK, but be sure to make sure that these multiple things can indeed be regarded as the same type of thing or are particularly similar in semantics.
    """
    message = [
        {
            "role":"user",
            "content":[
                {
                    "type":"text",
                    "text":userprompt,
                },
                {
                    "type":"image",
                    "image":image
                }
            ]
        },
        {
            "role":"system",
            "content":[
                {
                    "type":"text",
                    "text":systemprompt,
                },
                {
                    "type":"image",
                    "image":resize_image
                }
            ]
        }
    ]
    
    output = useqwen(message)
    answer,data,bbox = extract_bbox(output)
    return bbox

def force_findbox(subject, image):
    """_summary_

    Args:
        subject (_type_): _description_
        comimage (_type_): _description_
        image (_type_): _description_


    To prevent any failure in search or VLM box finding, we use this function to directly find box from image and subject.

    """
    
    outformat = {
        "answer": "NO",
        "box": [[-1, -1, -1, -1]],
        "center point": [-1, -1],
        "image file name": "~~~"
    }
    
    system_prompt = f"""
    You is a smart subjects finder who can find the boxes of the subjects in the image.
    The subject you need to find is **{subject}**.
    You need to find the subjects in the image.
    Output in json format.
    If you can not find the subject,put "NO" in first "answer", output all -1 in boxes[0][bbox_2d] as [-1,-1,-1,-1].
    """
    
    user_prompt = f"""
    Please find the box of the {subject} in the image.
    ouptput in json format.
    """
    
    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_prompt,
                },
                {
                    "type": "image",
                    "image": image
                }
            ]
        },
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                }
            ]
        }
    ]
    output = useqwen(message)
    answer, data, bbox = extract_bbox(output)
    return bbox

from PIL import Image
import cv2
import numpy as np

def draw_boxes_cv2(img: np.ndarray,
                   boxes_with_color: dict,
                   output_path: str = None) -> np.ndarray:
    """
    Draw colored boxes and labels on a cv2 image (BGR).
    """
    annotated = img.copy()
    for label, info in boxes_with_color.items():
        coords = info['box']
        color  = info['color']  # BGR order
        if isinstance(coords[0], int):
            coords = [coords]
        for x1, y1, x2, y2 in coords:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    if output_path:
        cv2.imwrite(output_path, annotated)
        print(f"✅ Annotated image saved to: {output_path}")
    return annotated

if __name__ == "__main__":
    # test_image = "path/to/your/testcode/disktest.jpg"
    test_image = "path/to/your/LIBERO/test/videos/first_frame.jpg"
    #mainsubjects = ["Juice bottle","school bag","laptop","green tennis ball"]
    mainsubjects = ["block bowl","plate"]
    resize_image,scale,(w,h)= load_and_resize(test_image,max_side=512)
    boxes = from_sub_to_box(mainsubjects, resize_image) ##the image在 never resize
    print(boxes)
    cv2_resize_image = cv2.cvtColor(np.array(resize_image), cv2.COLOR_RGB2BGR)
    annotated = draw_boxes_cv2(
        cv2_resize_image,
        boxes,
        output_path='path/to/your/script/test/libero_test.jpg',
    )
    
    