
"""_summary_

    NOTES:
    starting steps:
    1. zsh path/to/your/experiments/robot/libero_run/mps_start.sh
    2. conda activate path/to/your/conda_env/ftdino
        export TRANSFORMERS_OFFLINE=1
        export HF_HUB_OFFLINE=1
        python path/to/your/experiments/robot/libero_run/vision_planner_service.py \
        --endpoint ipc:///tmp/vision_planner.sock --device cuda:0
    3. start this script
        conda activate path/to/your/conda_env/openvla
        python path/to/your/experiments/robot/libero_run/main_agent_clean.py
            --pretrained_checkpoint path/to/your/openvla/pretrained_checkpoint
            --task_suite_name libero_task_name
    4. zsh path/to/your/experiments/robot/libero_run/mps_stop.sh
"""

# ====== VisionPlannerClient (main process) ======
try:
    import cv2
except ImportError as e:
    print(e)
import zmq, io, json, pickle
from PIL import Image
import numpy as np
_np = np
from collections import deque
import logging
logging.getLogger("inference_core").setLevel(logging.ERROR)

class VisionPlannerClient:
    def __init__(self, endpoint="ipc:///tmp/vision_planner.sock", timeout_ms=600000):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.sock.connect(endpoint)

    @staticmethod
    def _img_to_jpg_bytes(arr_or_pil, q: int = 85) -> bytes:
        if isinstance(arr_or_pil, np.ndarray):
            im = Image.fromarray(arr_or_pil.astype(np.uint8)).convert("RGB")
        else:
            im = arr_or_pil.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=q)
        return buf.getvalue()

    def plan(self, task: str, init_image: np.ndarray):
        header = {"cmd": "plan", "task": task}
        jpg = self._img_to_jpg_bytes(init_image)
        self.sock.send_multipart([json.dumps(header).encode("utf-8"), jpg])
        parts = self.sock.recv_multipart()
        body = json.loads(parts[0].decode("utf-8"))
        if not body.get("ok", False):
            raise RuntimeError(f"plan failed: {body}")
        return body["plan"], body["objs"], body["locas"]

    def detect(self, first_frame: np.ndarray, obj_prompts, loc_prompts, pred_thr=0.3, web=True):
        header = {"cmd": "detect", "obj_prompts": obj_prompts, "loc_prompts": loc_prompts, "pred_thr": float(pred_thr), "web": web}
        jpg = self._img_to_jpg_bytes(first_frame)
        self.sock.send_multipart([json.dumps(header).encode("utf-8"), jpg])
        parts = self.sock.recv_multipart()
        body = json.loads(parts[0].decode("utf-8"))
        if not body.get("ok", False):
            raise RuntimeError(f"detect failed: {body}")
        dirt = pickle.loads(parts[1])
        replace = pickle.loads(parts[2])
        return dirt, replace

    def verify(self, image_pair_queue: deque, instruction: str, max_pairs: int = 10) -> bool:
        """Verify using a queue of image pairs.

        image_pair_queue: deque[(PIL.Image, PIL.Image)] or [(np.ndarray, np.ndarray)].
        Take the last `max_pairs` pairs and send them in order (main_img, eih_img).
        """
        header = {"cmd": "verify", "instruction": instruction}
        pairs = list(image_pair_queue)[-max_pairs:]
        blobs = [json.dumps(header).encode("utf-8")]
        for (main_img, eih_img) in pairs:
            blobs.append(self._img_to_jpg_bytes(main_img))
            blobs.append(self._img_to_jpg_bytes(eih_img))
        self.sock.send_multipart(blobs)
        parts = self.sock.recv_multipart()
        body = json.loads(parts[0].decode("utf-8"))
        if not body.get("ok", False):
            raise RuntimeError(f"verify failed: {body}")
        return bool(body["complete"])

import os,re
import sys
### sys.excepthook = sys.__excepthook__   ## only for debug
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import tqdm
sys.path.append("path/to/your/LIBERO_ZERO/")
from libero.libero import benchmark

import swanlab
import collections
import time
import sys
# timing accumulators (global, non-invasive)
TIMING_ACC = {
    'planner': [],
    'vision': [],
    'VOC': [],
    'VOC_init': [],
    'VLA': [],
    'VLM': []
}
sys.path.append("path/to/your/VLA-2")
# from script.auto_DL import initGLMT
# from script.Wholebody import goforward
# from script.mmgdino import mmgdino, init_detector,
from script.mmgdino_simple import inject_colors_into_tasklist,make_instruction,check_stuck_by_actions
from script.segvideo_simple import init_models_and_params,init_processor_first_step_v2,track_frame_and_overlay


# Append current directory so that interpreter can find experiments.robot
sys.path.append("path/to/your/VLA-2/")
from experiments.robot.libero_run.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
    resize_image, 
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


# from qwenvl import load_qwen_vl_model, check_completion_with_qwen_vl
VLM_AVAILABLE = True


@dataclass
class GenerateConfig:

    # fmt: off

    # #################################################################################################################
    # # Model-specific parameters
    # #################################################################################################################
    model_family: str = "openvla"                # Model family
    pretrained_checkpoint: Union[str, Path] = "path/to/your/openvla/pretrained_checkpoint"     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    # #################################################################################################################
    # # LIBERO environment-specific parameters
    # #################################################################################################################
    task_suite_name: str = "libero_goal"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 5                      # Number of rollouts per task (original default 50)

    # #################################################################################################################
    # # Utils
    # #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "path/to/your/experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "VLA-2"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "agent_eval"          # Name of entity to log under

    seed: int = 42                                    # Random Seed (for reproducibility)

    save_frames: bool = False
    frames_save_root_dir: str = "path/to/your/experiments/saved_frames"
    use_verifier: str = True
    verify_frequency: int = 20
    replace: bool = True

    thr: float = 0.3
    # fmt: on


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    # if task_suite_name contains libero_*，such as "libero_xxx" as unnorm_key（eg. "libero_10_no_noops" -> "libero_10"）
    if cfg.task_suite_name == "libero_0" :
        cfg.unnorm_key ="libero_goal"
    elif isinstance(cfg.task_suite_name, str) and cfg.task_suite_name.startswith("libero"):
        m = re.match(r'(libero(?:_[^_]+)?)', cfg.task_suite_name)
        cfg.unnorm_key = m.group(1) if m else cfg.task_suite_name
    else:
        cfg.unnorm_key = cfg.task_suite_name

    # Load VLA model (Executor E)
    vla_model = get_model(cfg) # Renamed for clarity

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        if cfg.unnorm_key not in vla_model.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla_model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in vla_model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor for VLA
    vla_processor = None
    if cfg.model_family == "openvla":
        vla_processor = get_processor(cfg) # Renamed for clarity

    if cfg.use_verifier:
    # Load VLM model (Verifier V) - Load once before the loops
    # vlm_model, vlm_processor = load_qwen_vl_model(model_id="path/to/your/qwenft/merged", device="cuda:0")
        VLM_ENABLED_EFFECTIVELY = True
    else:
        VLM_ENABLED_EFFECTIVELY = False
        print("No VLM")
    hardcoded_plan = [
        "put both the alphabet soup and the tomato sauce in the basket",
    ]
    
    print(f"Testing in hardcoded plan: {hardcoded_plan}")

    # Initialize SAM/Cutie models and other vision components
    print("Initializing SAM Cutie \n")
    SAM, Cutie, device = init_models_and_params()
    print("Initialized vision local model \n")
    planner = VisionPlannerClient(endpoint="ipc:///tmp/vision_planner.sock", timeout_ms=6000000)
    print("Finished initializing vision planner client.")
    # mmgdino_model = init_detector()
    # print("Initializing GLM and testing \n")
    # GLM_pro,GLM_mod = initGLMT()
    # tasklist,objs,locas = goforward(hardcoded_plan[0],None,GLM_pro,GLM_mod)
    # print(f"Based task:{hardcoded_plan}, tasklist: {tasklist}, objs: {objs}, locas: {locas}")
    print("All new model init finish. \n")

    # Initialize local logging
    run_id = f"PEV_V1-EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}" # Added PEV_V1 prefix
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")
    log_file.write(f"PEV V1 Evaluation Run\n")
    log_file.write(f"Hardcoded Plan: {hardcoded_plan}\n")
    log_file.write(f"VLM Verification Enabled: {VLM_ENABLED_EFFECTIVELY}\n")

    frames_run_dir = None # Initialize here
    if cfg.save_frames:
        frames_run_dir = Path(cfg.frames_save_root_dir) / run_id
        os.makedirs(frames_run_dir, exist_ok=True)
        print(f"Will save frames to subdirectories within: {frames_run_dir}")
        log_file.write(f"Saving frames enabled, base directory: {frames_run_dir}\n")

    # Initialize Weights & Biases (swanlab) logging as well
    if cfg.use_wandb:
        swanlab.init(
            project=cfg.wandb_entity,       # swanlab project name
            experiment_name=run_id,          # swanlab experiment name
            config=draccus.encode(cfg),      # record hyperparameters
        )
        # 追加额外字段
        swanlab.config.update({
            "hardcoded_plan": hardcoded_plan,
            "vlm_enabled": VLM_ENABLED_EFFECTIVELY,
        })


    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc="Tasks"):
        # Get task
        # task_id = 9
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and ORIGINAL task description
        env, original_task_description = get_libero_env(task, cfg.model_family, resolution=256)
        
        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task), desc=f"Task {task_id} Episodes", leave=False):
            print(f"\nTask: {original_task_description}")
            log_file.write(f"\nOriginal Task: {original_task_description}\n")

            # Reset environment
            env.reset()

            # Set initial state
            obs = env.set_init_state(initial_states[episode_idx])
            initimage = get_libero_image(obs,resize_size)

            # Make sublist and detect items in tasks
            start = time.perf_counter()
            hardcoded_plan, objs, locas = planner.plan(original_task_description, initimage)
            end = time.perf_counter()
            planner_time = end - start
            # print(f"Timer, planner, {planner_time:.3f}")

            current_subtask_index = 0
            if not hardcoded_plan:
                print("Warning: Plan is empty, cannot execute subtasks.")
                log_file.write("Warning: Plan is empty.\n")
                current_subtask_instruction = original_task_description # Fallback
            else:
                current_subtask_instruction = hardcoded_plan[current_subtask_index]
                # current_subtask_instruction = original_task_description
            print(f"Starting Episode {episode_idx+1} with Subtask 1: '{current_subtask_instruction}'")
            log_file.write(f"Episode {episode_idx+1} | Subtask {current_subtask_index+1}: {current_subtask_instruction}\n")
            plan_completed_by_vlm = False

            image_pair_queue = collections.deque(maxlen=10)

            episode_frame_save_dir = cfg.frames_save_root_dir
            if cfg.save_frames and frames_run_dir: # Check frames_run_dir exists
                episode_frame_save_dir = frames_run_dir / f"task_{task_id}" / f"episode_{episode_idx}"
                os.makedirs(episode_frame_save_dir, exist_ok=True)
                # log_file.write(f"Saving episode frames to: {episode_frame_save_dir}\n") # Maybe too verbose

            # Setup
            t = 0
            replay_images = []
            # ... (max_steps calculation remains the same) ...
            if cfg.unnorm_key == "libero_spatial" or "libero_spatial" in cfg.unnorm_key:
                max_steps = 220
            elif cfg.unnorm_key == "libero_object" or "libero_object" in cfg.unnorm_key:
                max_steps = 280
            elif cfg.unnorm_key == "libero_goal" or "libero_goal" in cfg.unnorm_key:
                max_steps = 300
            elif cfg.unnorm_key == "libero_10" or "libero_10" in cfg.unnorm_key: #520
                max_steps = 520
            elif cfg.unnorm_key == "libero_90" or "libero_90" in cfg.unnorm_key:
                max_steps = 400
            else: # Fallback
                max_steps = 400
                
            # recovering setting
            stop_time = 0
            recovering = False
            
            VLA_time_counter = 0
            verifier_time_counter = 0
            VOC_init_time = 0
            VOC_time = 0
            action = [0,0,0,0,0,0,0]
            action_history = collections.deque(maxlen=10)

            log_file.write(f"Starting episode run (max_steps={max_steps})...\n")
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # Wait steps
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))

                        if VLM_ENABLED_EFFECTIVELY and (t % 10 == 0): 
                            img_main_np_wait = get_libero_image(obs, resize_size)
                            img_eih_np_wait = obs["robot0_eye_in_hand_image"]
                            main_img_pil_wait = Image.fromarray(img_main_np_wait)
                            eih_img_pil_wait = Image.fromarray(img_eih_np_wait)
                            image_pair_queue.append((main_img_pil_wait, eih_img_pil_wait))
                    
                        t += 1
                        continue
                    
                    ### Make CV post adjustments
                    if t ==  cfg.num_steps_wait:
                        img_main_np_st = get_libero_image(obs, resize_size)
                        vision_start = time.perf_counter()
                        dirt, replace = planner.detect(img_main_np_st, objs, locas, pred_thr=cfg.thr)
                        vision_end = time.perf_counter()
                        vision_time = vision_end - vision_start
                        #print(f"Timer, vision, {vision_time:.3f}")
                        
                        VOC_start = time.perf_counter()
                        Cutie_processor,palette,objects = init_processor_first_step_v2(sam_model=SAM, processor=Cutie, first_frame=img_main_np_st, dirt = dirt, device = device)
                        VOC_end = time.perf_counter()
                        VOC_init_time = VOC_end - VOC_start
                        VOC_time += VOC_init_time
                        try:
                            hardcoded_plan,rep = inject_colors_into_tasklist(tasklist=hardcoded_plan, dirt=dirt,replace_or_not=cfg.replace,replace=replace) # org_task: original_task_description
                            current_subtask_instruction = hardcoded_plan[current_subtask_index]
                            if not cfg.use_verifier:
                                current_subtask_instruction = ','.join(hardcoded_plan)  ## use full sequence test
                            print(f"Replace color into tasklist success.\n report:{rep}")
                        except:
                            pass

                    # --- Get Images (Main and Eye-in-Hand) ---
                    # Get preprocessed main image (likely returns NumPy)
                    img_main_np_plane = get_libero_image(obs, resize_size)
                    
                    VOC_start = time.perf_counter()
                    img_main_np = track_frame_and_overlay(frame = img_main_np_plane,processor = Cutie_processor, palette = palette,alpha=0.3)
                    VOC_end = time.perf_counter()
                    VOC_time = VOC_end - VOC_start + VOC_time
                    
                    
                    # Get eye-in-hand image (NumPy)
                    img_eih_np = obs["robot0_eye_in_hand_image"]
                    # Save a temporary debug image to your local cache (change path as needed)
                    Image.fromarray(img_main_np).save("path/to/your/experiments/cache/image_flow/eyeinhand_temp.png")  # For debugging
                    # --- Save Frames (Every 20 steps, if enabled) ---
                    # Moved frame saving here to ensure we have the images before VLM check
                    if cfg.save_frames and episode_frame_save_dir and (t % cfg.verify_frequency == 0):
                        try:
                            # Save main image
                            frame_filename = f"frame_{t:04d}.png"
                            pil_img_main = Image.fromarray(img_main_np)
                            pil_img_main.save(episode_frame_save_dir / frame_filename)

                            # Save eye-in-hand image
                            frame_filename_eih = f"eyeinhand_{t:04d}.png"
                            pil_img_eih = Image.fromarray(img_eih_np)
                            pil_img_eih.save(episode_frame_save_dir / frame_filename_eih)
                        except Exception as save_e:
                            print(f"Warning: Failed to save frame {t}. Error: {save_e}")
                            log_file.write(f"Warning: Failed to save frame {t}. Error: {save_e}\n")

                
                    image_collected_this_step = False
                    if VLM_ENABLED_EFFECTIVELY and (t % 10 == 0): 
                        image_collected_this_step = True
                        
                        main_img_pil = Image.fromarray(img_main_np)
                        eih_img_pil = Image.fromarray(img_eih_np)
                        image_pair_queue.append((main_img_pil, eih_img_pil))

                        
                        if cfg.save_frames and episode_frame_save_dir:
                            try:
                                frame_filename = f"frame_{t:04d}.png"
                                main_img_pil.save(episode_frame_save_dir / frame_filename)
                                frame_filename_eih = f"eyeinhand_{t:04d}.png"
                                eih_img_pil.save(episode_frame_save_dir / frame_filename_eih)
                            except Exception as save_e:
                                print(f"Warning: Failed to save frame {t}. Error: {save_e}")
                                log_file.write(f"Warning: Failed to save frame {t}. Error: {save_e}\n")

                    
                    vlm_check_this_step = False
                    if VLM_ENABLED_EFFECTIVELY and (t % cfg.verify_frequency == 0): 
                        vlm_check_this_step = True

                    if vlm_check_this_step and not plan_completed_by_vlm and not recovering:
                        
                        verifier_start = time.perf_counter()
                        
                        print(f"[t={t}] Performing VLM check for subtask: '{current_subtask_instruction}' using queue (size {len(image_pair_queue)})")
                        log_file.write(f"[t={t}] VLM Check Start: '{current_subtask_instruction}' (Queue size: {len(image_pair_queue)})\n")

                        is_subtask_complete = planner.verify(
                            image_pair_queue=image_pair_queue,
                            instruction=current_subtask_instruction,
                            max_pairs=10
                        )

                        log_file.write(f"[t={t}] VLM Check Result: {'Complete' if is_subtask_complete else 'Not Complete'}\n")

                        if is_subtask_complete:
                            
                            print(f"[t={t}] VLM Confirmed: Subtask '{current_subtask_instruction}' COMPLETE.")
                            current_subtask_index += 1
                            if current_subtask_index < len(hardcoded_plan):
                                current_subtask_instruction = hardcoded_plan[current_subtask_index]
                                print(f"[t={t}] Switching to next subtask ({current_subtask_index + 1}/{len(hardcoded_plan)}): '{current_subtask_instruction}'")
                                log_file.write(f"[t={t}] VLM SWITCH -> Subtask {current_subtask_index+1}: {current_subtask_instruction}\n")
                            else:
                                print(f"[t={t}] VLM Confirmed: All {len(hardcoded_plan)} subtasks in the plan are complete.")
                                log_file.write(f"[t={t}] VLM: Plan finished.\n")
                                plan_completed_by_vlm = True
                                print(f"[t={t}] Continuing execution with last subtask instruction: '{current_subtask_instruction}' until env done or max_steps.")
                        verifier_end = time.perf_counter()
                        verifier_time = verifier_end - verifier_start
                        verifier_time_counter += verifier_time
                        
                    try:
                        action_history.append(_np.asarray(action, dtype=float))
                    except NameError:
                        action_history.append(np.asarray(action, dtype=float))

                    is_stuck, stuck_reason = check_stuck_by_actions( ### ！！！ this params may not work well in your project, we strongly recommend to test and adjust them
                        action_history,
                        min_len=8,               
                        trans_thresh=0.02,       
                        rot_thresh=0.03,
                        grip_change_thresh=0.1,
                        oscillation_flip_ratio=0.6
                    )

                    if is_stuck and not recovering:
                        old_task = current_subtask_instruction
                        current_subtask_instruction = "lift up the gripper"
                        stop_time = t + 10
                        recovering = True
                        print(f"[t={t}] STUCK detected by actions -> enter recovering mode. reason={stuck_reason}")
                        log_file.write(f"[t={t}] STUCK detected by actions. reason={stuck_reason}\n")


                    if t == stop_time:
                        if isinstance(old_task,str):
                            current_subtask_instruction = old_task
                        recovering = False
                        # Save main image for replay video (use the already processed img_main_np)
                    replay_images.append(img_main_np)

                    VLA_start = time.perf_counter()
                    # Prepare observations dict for VLA
                    observation = {
                        "full_image": img_main_np, # Use the main image
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }
                    arg_current_task_prompt = make_instruction(current_subtask_instruction,hardcoded_plan,simple=True)
                    action = get_action(
                        cfg,
                        vla_model, # Use VLA model
                        observation,
                        # Use the current subtask instruction, NOT original_task_description
                        arg_current_task_prompt,
                        vla_processor, # Use VLA processor
                    )
                    

                    # Normalize gripper action
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] Invert gripper action if needed
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)
                        # print(f"Action: {action}") # Can be verbose

                    VLA_end = time.perf_counter()
                    VLA_time = VLA_end - VLA_start
                    VLA_time_counter += VLA_time

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())

                    # --- Check Environment Done (Original Success Condition) ---
                    if done:
                        task_successes += 1
                        total_successes += 1
                        print(f"[t={t}] Environment signaled DONE (Success!).")
                        log_file.write(f"[t={t}] Environment DONE signal received.\n")
                        break # Break episode loop on success

                    t += 1 # Increment timestep

                    # Check for timeout
                    if t >= max_steps + cfg.num_steps_wait:
                         print(f"[t={t}] Episode TIMEOUT.")
                         log_file.write(f"[t={t}] Episode TIMEOUT.\n")
                         # Ensure 'done' is False if timeout occurred before env signaled success
                         done = False # Explicitly set done to False on timeout
                         break # Break episode loop on timeout


                except Exception as e:
                    print(f"Caught exception during episode execution: {e}")
                    log_file.write(f"Exception during episode: {e}\n")
                    import traceback
                    traceback.print_exc(file=log_file)
                    done = False # Assume failure on exception
                    break # Break episode loop on error

            # --- Episode End ---
            task_episodes += 1
            total_episodes += 1

            # Log timing info
            print(f"TIMER: Planner time {planner_time:.3f}s, Vision time {vision_time:.3f}s, VLA total {VLA_time_counter:.3f}s, VLM total {verifier_time_counter:.3f}s, VOC total {VOC_time:.3f}s, task info: {original_task_description}, episode {episode_idx+1}")
            # --- record timings to global accumulators (non-invasive) ---
            try:
                TIMING_ACC['planner'].append(float(planner_time))
            except Exception:
                pass
            try:
                TIMING_ACC['vision'].append(float(vision_time))
            except Exception:
                pass
            try:
                TIMING_ACC['VOC'].append(float(VOC_time))
            except Exception:
                pass
            try:
                TIMING_ACC['VOC_init'].append(float(VOC_init_time))
            except Exception:
                pass
            try:
                TIMING_ACC['VLA'].append(float(VLA_time_counter))
            except Exception:
                pass
            try:
                TIMING_ACC['VLM'].append(float(verifier_time_counter))
            except Exception:
                pass
            
            # Save a replay video of the episode
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=original_task_description, log_file=log_file
            )

            # Log current results
            print(f"Episode End. Success (based on env done signal): {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Episode End. Success (Env Done): {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

            # Log episode result to W&B if enabled
            if cfg.use_wandb:
                 swanlab.log({
                     f"episode_success/{original_task_description}": 1 if done else 0,
                     f"plan_completed_by_vlm/{original_task_description}": 1 if plan_completed_by_vlm else 0,
                     "step": total_episodes # Log against total episodes
                 })


        # --- Task End ---
        # Log final task results
        task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
        print(f"Task {task_id} ('{original_task_description}') finished. Task Success Rate: {task_success_rate:.2f} ({task_successes}/{task_episodes})")
        log_file.write(f"\nTask {task_id} Finished. Task Success Rate: {task_success_rate:.2f} ({task_successes}/{task_episodes})\n")

        # Log task summary to W&B
        if cfg.use_wandb:
            swanlab.log({
                 f"task_summary/success_rate": task_success_rate,
                 f"task_summary/num_successes": task_successes,
                 f"task_summary/num_episodes": task_episodes,
                 # Log against task_id or a global step if preferred
            }, step=task_id) # Log summary per task_id


    # --- Evaluation End ---
    # Log final total results
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    print(f"\nEvaluation Finished.")
    print(f"Total Success Rate: {total_success_rate:.2f} ({total_successes}/{total_episodes})")
    log_file.write(f"\nEvaluation Finished.\n")
    log_file.write(f"Total Success Rate: {total_success_rate:.2f} ({total_successes}/{total_episodes})\n")
    log_file.flush()

    # --- Print overall average timings collected during run ---
    def _avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    print("\nOverall timing averages (seconds):")
    print(f"  planner: {_avg(TIMING_ACC['planner']):.3f} | n={len(TIMING_ACC['planner'])}")
    print(f"  vision:  {_avg(TIMING_ACC['vision']):.3f} | n={len(TIMING_ACC['vision'])}")
    print(f"  VOC_init: {_avg(TIMING_ACC['VOC_init']):.3f} | n={len(TIMING_ACC['VOC_init'])}")
    print(f"  VOC_total: {_avg(TIMING_ACC['VOC']):.3f} | n={len(TIMING_ACC['VOC'])}")
    print(f"  VLA_total: {_avg(TIMING_ACC['VLA']):.3f} | n={len(TIMING_ACC['VLA'])}")
    print(f"  VLM_total: {_avg(TIMING_ACC['VLM']):.3f} | n={len(TIMING_ACC['VLM'])}")

    # also write to log file
    log_file.write("\nOverall timing averages (seconds):\n")
    log_file.write(f"  planner: {_avg(TIMING_ACC['planner']):.3f} | n={len(TIMING_ACC['planner'])}\n")
    log_file.write(f"  vision&language:  {_avg(TIMING_ACC['vision']):.3f} | n={len(TIMING_ACC['vision'])}\n")
    log_file.write(f"  VOC_init: {_avg(TIMING_ACC['VOC_init']):.3f} | n={len(TIMING_ACC['VOC_init'])}\n")
    log_file.write(f"  VOC_total: {_avg(TIMING_ACC['VOC']):.3f} | n={len(TIMING_ACC['VOC'])}\n")
    log_file.write(f"  VLA_total: {_avg(TIMING_ACC['VLA']):.3f} | n={len(TIMING_ACC['VLA'])}\n")
    log_file.write(f"  VLM_total: {_avg(TIMING_ACC['VLM']):.3f} | n={len(TIMING_ACC['VLM'])}\n")
    log_file.flush()

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        swanlab.log(
            {
                "total_summary/success_rate": total_success_rate,
                "total_summary/num_successes": total_successes,
                "total_summary/num_episodes": total_episodes,
            }
        )
        swanlab.save(local_log_filepath)
        swanlab.finish()


if __name__ == "__main__":
    
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.perf_counter() 

    eval_libero()

    end_time = time.perf_counter() 
    duration = end_time - start_time
    print(f"\n end: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"eval_libero total_time: {duration:.2f} seconds")
