# qwenvl.py (fixed - image and text input Prompt)

import torch,sys
from PIL import Image
import numpy as np
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import traceback
import collections 
sys.path.append("/zhaohan/ZJX/Agentic_VLA/code/script/")
from qwenvl_meg import enrich_subtask_v2


from qwen_vl_utils import process_vision_info
QWEN_UTILS_AVAILABLE = True
print("Successfully imported qwen_vl_utils.process_vision_info.")


def load_qwen_vl_model(model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: str = "auto"):

    print(f"Attempting to load VLM model: {model_id} to device: {device} (using Flash Attention 2)")
    model = None
    processor = None
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" # eager or flash_attention_2
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print("VLM model and processor loaded successfully.")
        return model, processor
    except Exception as e:
        print(f"Error: Failed to load VLM model/processor: {e}")
        traceback.print_exc()
        return None, None


def check_completion_with_qwen_vl(
    vlm_model,
    vlm_processor,
  
    image_pair_queue: collections.deque,  

    current_subtask_instruction: str
) -> bool:
    """
    Uses the loaded Qwen-VL model to check if the current subtask is complete,
    based on a sequence of image pairs.
    Relies on zero-shot prompting and parsing of "Yes"/"No" responses.
    Uses qwen_vl_utils.process_vision_info for image processing.
    Prompts and parsing are in English.
    """

    if vlm_model is None or vlm_processor is None:
        print("[VLM Check] Error: VLM model or processor not loaded.")
        return False
    if not QWEN_UTILS_AVAILABLE:
        print("[VLM Check] Error: qwen_vl_utils not available.")
        return False
    
    if not image_pair_queue:
        print("[VLM Check] Warning: Image queue is empty. Cannot perform check.")
        return False
    

    try:
        messages = enrich_subtask_v2(current_subtask_instruction, image_pair_queue)
        text_template = vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages,return_video_kwargs=True)
        if image_inputs is None and video_inputs is None:
             print("[VLM Check] Error: process_vision_info failed to process image inputs.")
             return False

        inputs = vlm_processor(
            text=[text_template],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(vlm_model.device)

        with torch.no_grad():
            generated_ids = vlm_model.generate(
                **inputs,
                max_new_tokens=10, # Still expect short Yes/No
                min_new_tokens=3,
                do_sample=False,
                pad_token_id=vlm_processor.tokenizer.eos_token_id # Corrected access
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = vlm_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        if response:
            vlm_response_text = response[0].strip()
            # print(f"[VLM Check] VLM raw response: '{vlm_response_text}'") # Debug

            # Be a bit lenient: check lower case and potential starting variations
            if "yes" in vlm_response_text.lower() and "no" not in vlm_response_text.lower():
                print(f"[VLM Check] VLM Judgement: COMPLETED (Response: '{vlm_response_text}')")
                return True
            else:
                print(f"[VLM Check] VLM Judgement: Not Completed (Response: '{vlm_response_text}')") # Debug
                return False

        else:
            print("[VLM Check] VLM Judgement: Not Completed (No valid response)") # Debug
            return False

    except Exception as e:
        print(f"Error: Exception during VLM check (Instruction: '{current_subtask_instruction}'): {e}")
        traceback.print_exc()
        return False