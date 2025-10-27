"""
Forward pipeline:
Input: image stream and a task description.
Use SAP to split the task, use judge to generate boxes, and use auto_DL as a knowledge helper.
Produce box data and subtask segmentation results.
Pass box data to Cutie+SAM to generate segmentations.
Loop until the task completes or an error occurs.

Feedback:
Use a verifier to check whether a subtask is completed. If completed, move to the next subtask and refresh boxes; otherwise continue execution.
"""
from anyio import Path
import cv2,sys
sys.path.append("path/to/your/script")
from SAPdivision import SAP
# from Judge_simple import from_sub_to_box

def goforward(task, firstimage,processor_GLM,model_GLM):
    """_summary_

    输入是第一个图像和任务描述
    
    输出是一个包含两个列表的元组，第一个列表是物体的box和颜色，第二个列表是位置的box和颜色
    和一个任务清单，格式：
    tasklist[1] = "do this"
    tasklist[2] = "do that"

    input: task:str,firstimage:np.ndarray
    output: (tasklist: list, objs: list, location: list)
    """
    if firstimage is None or not hasattr(firstimage, 'any') or not firstimage.any():
        # If no initial image is provided, try to read the first frame from a local test video (change path as needed)
        cap = cv2.VideoCapture("path/to/your/LIBERO/datasets/sample_video.mp4")
        _,firstimage = cap.read()
        cap.release()
    sublist = SAP(task, firstimage,processor_GLM,model_GLM) 
    tasklist = [item['subtask'] for item in sublist if isinstance(item, dict) and 'subtask' in item]
    objs = []
    location = []

    for obj in sublist:
        obj = obj["objects"]
        if len(obj) > 1:
            # check obj[-1] in location 
            if obj[-1] not in location:
                location.append(obj[-1])
            # check obj[:-1] in objs
            for single_obj in obj[:-1]:
                if single_obj not in objs:
                    objs.append(single_obj)
        else:
            # check obj[0] in objs
            if obj[0] not in objs:
                objs.append(obj[0])

    objs = [obj for obj in objs if obj not in location]
    return tasklist, objs, location


if __name__ == "__main__":
    video = "path/to/your/testcode/testimage/movegreen.mp4"
    task = "place the tennis ball on the book"

    
    # Read the first frame of the video
    cap = cv2.VideoCapture(video)
    ret, firstimage = cap.read()
    cap.release()

    if not ret:
        raise ValueError("Cannot read the first frame from the video. Check the video path and file.")
    tasklist, objs, location = goforward(task, firstimage)
    
    print("Task List:", tasklist)
    print("Objects:", objs)
    print("Location:", location)
