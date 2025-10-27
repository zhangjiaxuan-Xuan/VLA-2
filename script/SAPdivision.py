from auto_DL import useqwen,useglmt,initGLMT
import base64
from PIL import Image
import io
import numpy as np

###
# NOTE: SAP logic currently focuses on pick-and-place style tasks.
# Consider generalizing and cleaning the API for broader task types.

###

def SAPdivsion(
    task_description: str,
    image:any,
    sign: str,
    processor_GLM,model_GLM
    ) :

    if sign!="success":
        if sign=="no subtask found":
            additional_info = "PAY MORE ATTENTION TO THE SUBTASKS in your last output, no valid subtask found. You should output the subtask in the same format as the example, without any other analysis or description."
        elif sign=="no objects found":
            additional_info = "PAY MORE ATTENTION TO THE OBJECTS in your last output, no valid objects found in /(here)/. You should output the objects in the same format as the example, without any other analysis or description."
        else:
            additional_info = "PAY MORE ATTENTION TO THE SUBTASKS and OBJECTS in your last output, no valid subtask or objects found. You should output the subtask and objects in the same format as the example, without any other analysis or description."
    else:
        additional_info = "You are doing a good job, keep it up"
    outputformat = {
        "subtask": [],
        "target objects":[],
        "is this subtask done?": []
    }

    inlist = "white mug, yellow and white mug, plate, basket, black bowl, butter, cream cheese box, alphabet soup, ketchup, tomato sauce, chocolate pudding, milk, orange juice, bbq sauce, salad dressing, cabinet, rack, moka pot, stove, microwave, wine bottle, caddy, book"

    task_decomposition_prompt =f"""
    You are a planning assistant for a fixed robotic arm. Your goal is to break down a high-level task into a sequence of **essential high-level commands**, suitable for a capable Vision-Language-Action (VLA) model to execute directly.

    Output Format:
    Generate a numbered list of commands. Each command should represent a significant action achieving a clear sub-goal. Stick to the allowed high-level actions.

    Example Plan Format (Use **exactly** this level of granularity):
    Plan for the robot arm:

    Goal: <original instruction>
    1. pick up the <object_name_1> /(<object_name_1>)/
    2. place the <object_name_1> in the <target_location> /(<object_name_1>,<target_location>)/
    3. pick up the <object_name_2> /(<object_name_2>)/
    4. place the <object_name_2> in the <target_location> /(<object_name_2>,<target_location>)/
    
    --- Example for a different task ---
    Goal: Put the apple in the red bowl
    1. pick up the apple /(apple)/
    2. place the apple in the red bowl /(apple,red bowl)/
    
    --- Example for another task ---
    Goal: Put the cup in the microwave and close it
    1. pick up the cup /(cup)/
    2. place the cup in the microwave /(cup,microwave)/
    3. close the microwave /(microwave)/
    
    --- Example for another task ---
    Goal: Turn on the stove and put the pot on it
    1. turn on the stove /(stove)/
    2. pick up the pot /(pot)/
    3. place the pot on the stove /(pot,stove)/
    
    --- Example for another task ---
    Goal: Put both books on the bookshelf
    1. pick up the red book /(red book)/
    2. place the red book on the bookshelf /(red book, bookshelf)/
    3. pick up the brown book /(brown book)/
    4. place the brown book on the bookshelf /(brown book, bookshelf)/
    
    --- Example for another task ---
    Goal: pick the red book near the butter and the brown book on the plate and put them on the left bookshelf
    1. pick up the red book near the butter /(red book)/
    2. place the red book near the butter on the left bookshelf /(red book, bookshelf)/
    3. pick up the brown book on the plate /(brown book)/
    4. place the brown book on the plate on the left bookshelf /(brown book, bookshelf)/
    
    --- Example for another task ---
    Goal: pick up the yellow and white mug next to the cookie box and place it on the plate
    1. pick up the yellow and white mug next to the cookie box /(yellow and white mug)/
    2. place the yellow and white mug next to the cookie box on the plate /(yellow and white mug, plate)/

    --- Example for another task ---
    Goal: put the black bowl in the bottom drawer of the cabinet and close it
    1. pick up the black bowl /(black bowl)/
    2. place the black bowl in the bottom drawer of the cabinet /(black bowl, cabinet)/
    3. close the bottom drawer of the cabinet /(cabinet)/

    Instructions:
    - Generate **only** high-level commands. 
    - You output should in the ***ABSOLUTELY SAME format*** as the example above. Even with unseen tasks, follow the same structure. ***WITHOUT ANY OTHER ANALYSISand DESCRIPTION***.
    - **After each command**, include a comment with the object names and locations in */()/*, this is nessary for the VLA model to understand which objects are involved in each command.
    - DO NOT include any descriptions of position and order in */()/* (e.g., "first pot", "back of the shelf","bottom of sth","upper of sth"), only color and shape are permitted (e.g., "red bowl", "cylindrical box").
        But you should maintain the details of the objects and locations as described in the task to subtask such as "red bowl near the plate", "brown book on the cabinet", "left bookshelf", "black bowl next to the cookie box", etc.
    - **ONLY USE */()/* to EXPRESS *OBJECTS*.** Comments, explanations, and anything else that has nothing to do with expressing objects are not allowed.
    - When an object or location has a qualifying modifier, such as a cabinet's drawer, door of a microwave, or the handle of pot , what you are expected to display in the /()/ is actually the **largest specific items these expressions** refer to, which are cabinets, microwaves, and pots, not the parts or subordinate items on these items that belong to these items.
        Meanwhile you should still maintain the detail expression in the subtask as "the drawer of the cabinet","the door of the microwave" (eg. pick up the bottle on the stove; pick up the bowl in the drawer).
    - **Allowed commands are strictly limited to:**
        - `pick up [object]`
        - `place [object] on [location]`
        - `place [object] in [location]`
        - `open [object/container/drawer/cabenit/etc.]`
        - `close [object/container/drawer/cabenit/etc.]`
        - `turn on [device]`
        - `turn off [device]`
    - Use the commands above **only when necessary** to achieve the goal. Most tasks will primarily use `pick up` and `place`.
    - **Explicitly DO NOT include separate steps for:**
        - `locate` (Assume VLA finds the object as part of executing the command)
        - `move to` or `move towards` (Assume the command includes necessary travel)
        - `lift`, `lower`, `grasp`, `release`, `push`, `pull`, `rotate`, `adjust` (Assume high-level commands handle these internally)
    - **Assume the VLA model handles all implicit actions:**
        - "pick up [object]" means: Find the object, navigate to it, grasp it securely, and lift it.
        - "place [object] in [location]" means: Transport the object to the location, position it correctly, and release the grasp.
        - "open/close [container]" means: Find the handle/seam, interact with it appropriately (pull, slide, lift) to change the container's state.
        - "turn on/off [device]" means: Find the correct button/switch, interact with it to change the device's power state.
    - Use the descriptive names from the task description and **DO NOT make any distortions** in subtasks (e.g.,if task involves {inlist}, make sure the subtasks about them are exactly the same).
    - Generate the minimal sequence of these high-level commands required to fulfill the Goal. Ensure the sequence logically achieves the task (e.g., you might need to `open` a drawer before `place`ing something inside it, even if 'open' isn't explicitly stated in the goal).
    - Additional INFO:{additional_info}
    Task: {task_description}
    Output:
    """
    if (not isinstance(image, str) and hasattr(image, 'size') and image.size == 0) or (isinstance(image, str) and image == "None"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": task_description},
                ],
            },
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": task_decomposition_prompt,
                    }
                ]
            }
        ]
    else:
        if isinstance(image, str):  # If the image is a file path
            try:
                with open(image, "rb") as img_file:
                    encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
            except Exception as e:
                encoded_string = None  # Handle the error as needed
        elif isinstance(image, Image.Image):  # If the image is a PIL Image
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")  # You can change the format if needed
            encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
        elif isinstance(image, np.ndarray):  # If the image is a NumPy array
            pil_image = Image.fromarray(image)
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")  # You can change the format if needed
            encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
        else:
            encoded_string = None  # Handle unsupported formats

        # Now you can use encoded_string in your messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": encoded_string,
                    },
                    {"type": "text", "text": task_description},
                ],
            },
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": task_decomposition_prompt,
                    }
                ]
            }
        ]
    output_text = useglmt(messages,processor_GLM,model_GLM)
    return output_text

from typing import List, Dict
import re

Task = Dict[str, object]

def parse_llm_plan_plus(raw_plan_string: str) -> List[Dict]:
    """
    Input example:
        1. pick up the apple / (apple)
        2. place the apple in/on the disk / (apple,disk)

    Output example:
        [
            {"subtask": "pick up the apple", "objects": ["apple"]},
            {"subtask": "place the apple in/on the disk", "objects": ["apple", "disk"]}
        ]
    Added logic:
        1. If subtask or objs is empty → return ["failure"]
        2. If objs≥3, deduplicate by "longest retention" until objs<3 or no more merging possible
    """
    DIRECTIONS = {
        "right", "left", "front", "back", "above", "below",
        "north", "south", "east", "west"
    }
    TRAILING_NUM_RE = re.compile(r"\s*\d+\s*$")
    PREPOSITIONS = {"in", "on", "into", "to", "onto", "inside", "outside"}

    if "\\n" in raw_plan_string:
        raw_plan_string = raw_plan_string.replace("\\n", "\n")

    line_re = re.compile(r"^\s*\d+\s*[).、.]?\s*(.+)$")
    obj_split_re = re.compile(r"[，,;；、]+")
    tasks: List[Dict] = []

    for raw_line in raw_plan_string.splitlines():
        m = line_re.match(raw_line)
        if not m:
            continue
        remainder = m.group(1).strip()

        # 1. Extract subtask
        sub_chars = []
        i = 0
        L = len(remainder)
        while i < L:
            ch = remainder[i]
            if ch == '/' or ch =='*':
                j = i - 1
                while j >= 0 and remainder[j] == ' ':
                    j -= 1
                word_end = j
                while j >= 0 and remainder[j].isalpha():
                    j -= 1
                word = remainder[j+1:word_end+1].lower()
                if word not in PREPOSITIONS:
                    break
            if ch in {'（', '<', '《', '"', "'"}:
                break
            sub_chars.append(ch)
            i += 1
        subtask = ''.join(sub_chars).strip()

        # 2. Extract objects
        left = max(remainder.rfind('('), remainder.rfind('（'))
        right = max(remainder.rfind(')'), remainder.rfind('）'))
        objs = []
        if 0 <= left < right:
            raw_objs = remainder[left+1:right]
            cleaned = re.sub(r"[<>\[\]{}\"\'（）()/]", " ", raw_objs)
            for o in obj_split_re.split(cleaned):
                o2 = o.strip()
                if not o2:
                    continue
                o3 = TRAILING_NUM_RE.sub("", o2).strip()
                low = o3.lower()
                if not low or low.isdigit() or low in DIRECTIONS:
                    continue
                objs.append(o3)

        # 4. Deduplicate by longest retention when objs ≥ 3
        while len(objs) >= 3:
            # word -> [(idx, len)]
            word_map = {}
            for idx, s in enumerate(objs):
                for w in set(s.lower().split()):
                    word_map.setdefault(w, []).append((idx, len(s)))
            # Find words that appear ≥ 2 times
            common = {w: lst for w, lst in word_map.items() if len(lst) >= 2}
            if not common:
                break
            to_del = set()
            for w, lst in common.items():
                keep_idx = max(lst, key=lambda x: x[1])[0]
                to_del.update(idx for idx, _ in lst if idx != keep_idx)
            for idx in sorted(to_del, reverse=True):
                del objs[idx]

        # 5. Still empty after deduplication
        if not subtask or all(not t.strip() for t in subtask):
            return "failure", "no subtask found"
        if not objs or all(not o.strip() for o in objs):
            sub_ds = decompose_liberotask(subtask)
            objs = sub_ds[0]["objects"]
            # Remove duplicates from objs while preserving order
            seen = set()
            unique_objs = []
            for o in objs:
                if o not in seen:
                    unique_objs.append(o)
                    seen.add(o)
            objs = unique_objs
            subtask = sub_ds["subtask"]
        if not objs or all(not o.strip() for o in objs):
            return "failure", "no objects found"
        # Assume subtask and objs are available at this point
        # First perform double space truncation filtering
        filtered_objs = []
        for obj in objs:
            if "  " in obj:
                # Truncate from the first consecutive double space, discard the rest
                obj = obj.split("  ", 1)[0]
            filtered_objs.append(obj)
        tasks.append({"subtask": subtask, "objects": filtered_objs})

    return tasks,"success"

from nltk.tokenize import word_tokenize
from nltk import pos_tag
import multiprocessing

def decompose_liberotask(task: str) -> dict:
    """
    Hard-coded decomposition of liberotask:
      - tasklist[0] = original task
      - objs = extract phrases after the first and last 'the'
        (phrase definition: consecutive JJ/NN type POS tags)
    
    Returns:
      {
        "tasklist": [ task ],
        "objs":      [ first_np, last_np ]  # at most two
      }
    """
    # 1. Tokenize and POS tagging
    tokens = word_tokenize(task)                        # tokenization
    tagged = pos_tag(tokens)                            # POS tagging

    # 2. Locate all positions of "the" (case-insensitive matching)
    the_idxs = [i for i, tok in enumerate(tokens) if tok.lower() == "the"]

    defs = []
    for idx in (the_idxs[:1] + the_idxs[-1:])[:2]:      # 取第一个、最后一个
        np_words = []
        # from "the" onwards, collect all adjectives and nouns
        for tok, tag in tagged[idx+1:]:
            if tag.startswith("JJ") or tag.startswith("NN") or tag.startswith("CC"):
                np_words.append(tok)
            else:
                break
        if np_words:
            defs.append(" ".join(np_words))
    cleaned = []
    for obj in defs:
        if "  " in obj:
            # take the part before the first double space
            obj = obj.split("  ", 1)[0]
        cleaned.append(obj)
    return [{
        "subtask": task,
        "objects": cleaned
    }]

def run_SAPdivision_with_timeout(task_description: str, image: any, sign: str,processor_GLM,model_GLM, timeout: int = 255) -> str:
    """
    using useqwen function SAPdivsion.
    input:
        task_description: high-level task description
        image: optional image input (if any)
        sign: status flag
        timeout: timeout duration (seconds)
    output:
        Decomposed subtask list and related information
    """
    def run_with_timeout(func, args=(), kwargs={}, timeout=255):
        """
        Run a function with a timeout.
        """
        pool = multiprocessing.Pool(processes=1)
        result = pool.apply_async(func, args, kwargs)
        try:
            return result.get(timeout)
        except multiprocessing.TimeoutError:
            pool.terminate()
            return "Timeout"

    try:
        output = run_with_timeout(SAPdivsion, args=(task_description, image, sign,processor_GLM,model_GLM), timeout=timeout)
        if output == "Timeout":
            print("SAP division timed out.")
            return "Timeout"
        return output
    except Exception as e:
        print(f"Error during SAP division: {e}")
        return "Timeout"

def SAP(task:str,image:any,processor_GLM,model_GLM):
    """
    SAP: Subtask Action Planning
    This function is used to decompose high-level tasks into a series of subtasks and actions, suitable for robots or automation systems.
    Input:
        task: high-level task description
        image: optional image input (if any)
    Output:
        Decomposed subtask list and related information
    """
    sign = "success"

    for _ in range(2): 
        # output = run_SAPdivision_with_timeout(task_description=task, image=image, sign=sign,processor_GLM = processor_GLM,model_GLM = model_GLM, timeout=315)
        output = SAPdivsion(task_description=task, image=image, sign=sign,processor_GLM = processor_GLM,model_GLM = model_GLM)
        if isinstance(output, str) and output == "Timeout":
            sublist = "failure"
            break
        if not output[0]=="<":
            match = output[0]
        else:
            try:
                match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL).group(1).strip()
                print(match)
            except Exception as e:
                print(f"Error occurred while extracting answer: {e}")
                sublist = "failure" 
                break
        try:
            sublist,sign = parse_llm_plan_plus(match)
        except Exception as e:
            print(f"Error occurred: {e}")
            sublist = "failure"
        if sublist == "failure" or sublist == []:
            print("Failed to parse the task. Retrying...")
            if sign=="no subtask found":
                print("No subtask found, retrying...")
                continue
            elif sign=="no objects found":
                print("No objects found, retrying...")
                continue
            else:
                print("Unknown error, retrying...")
                continue
        else:
            sign = "success"
            break
    try:
        if (
            (isinstance(sublist, str) and sublist == "failure")
            or (isinstance(sublist, list)
            and (
                not sublist                       # sublist is empty
                or not sublist[0].get("subtask") # subtask does not exist or is empty
                or not sublist[0].get("objects")     # objects does not exist or is empty
            ))
        ):
            try:
                sublist = decompose_liberotask(task)
            except Exception as e:
                sublist = {"subtask": task, "objects": []}
                print(f"Error occurred while decomposing task: {e}")
    except Exception as e:
        None
    return sublist

if __name__ == "__main__":
    # task_description="open the top drawer and put the bowl inside"
    task_description = "place the yellow and green ball on the Dream of Red Mansions book"
    image = "path/to/your/script/test/book_ball.jpg"
    # image="path/to/your/testimage_exp/kendrick_lamar_photos/attachment-Kendrick-Lamar.jpg"
    # image = "None"
    GLM_pro,GLM_model = initGLMT()
    output = SAPdivsion(task_description=task_description,image=image,sign="success",processor_GLM=GLM_pro,model_GLM=GLM_model)
    print(output)
    if not output[0]=="<":
        sentence = output[0]
    else:
        match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL).group(1).strip()
        print(match)
    print(parse_llm_plan_plus(match))