import torch
from PIL import Image
from numpy import asarray
import os
import numpy as np
import pandas as pd
from PIL import Image
import numpy as np
import transformers
import re
import sys
import os
from huggingface_hub import snapshot_download
from pathlib import Path

from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop

evaluation_dir = os.path.abspath(os.path.dirname(__file__))
radialog_path = os.path.join(evaluation_dir, "RaDialog")
if radialog_path not in sys.path:
    sys.path.append(radialog_path)

# radialog imports 
from LLAVA_Biovil.llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, remap_to_uint8
from LLAVA_Biovil.llava.model.builder import load_pretrained_model
from LLAVA_Biovil.llava.conversation import SeparatorStyle, conv_vicuna_v1
from LLAVA_Biovil.llava.constants import IMAGE_TOKEN_INDEX






def load_model_and_processor(model_name, device_map='cpu'):

    processor = None
    tokenizer = None
    
    if model_name == 'radialog':
        repo_id = "ChantalPellegrini/RaDialog-interactive-radiology-report-generation"

        model_path = snapshot_download(repo_id=repo_id, revision="main")
        model_path = Path(model_path)

        tokenizer, model, _, _ = load_pretrained_model(
            model_path, 
            model_base='liuhaotian/llava-v1.5-7b',
            model_name="llava-v1.5-7b-task-lora_radialog_instruct_llava_biovil_unfrozen_2e-5_5epochs_v5_checkpoint-21000", 
            load_8bit=False,
            device_map=device_map, 
            load_4bit=False
            )
        
    
    elif model_name == 'chexagent':
        model_id = "StanfordAIMI/CheXagent-2-3b"
        dtype = torch.bfloat16
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map, trust_remote_code=True)
        model = model.to(dtype)
        model.eval()
    elif model_name == 'llavamed':
        from radvlm.evaluation.llava_med_loading import register_llava_med_hf
        model_path = 'microsoft/llava-med-v1.5-mistral-7b'
        register_llava_med_hf()
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            local_files_only=False,
            device_map=device_map,
            trust_remote_code=True,
        )
        processor = transformers.AutoProcessor.from_pretrained(
            model_path, 
            local_files_only=False, 
            trust_remote_code=True
        )

    elif model_name == 'maira2':
        model = transformers.AutoModelForCausalLM.from_pretrained(
            'microsoft/maira-2',
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=device_map
            )
        processor = transformers.AutoProcessor.from_pretrained(
            'microsoft/maira-2', 
            trust_remote_code=True
            )
    elif model_name=='qwen2vl':
        model = transformers.Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.float16, device_map=device_map, attn_implementation="flash_attention_2"
            )
        processor = transformers.AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")


    else:
        # Load llava-ov checkpoint 
        common_kwargs = {
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True, 
        }

        if model_name == 'llavaov':
            model_name = 'llava-hf/llava-onevision-qwen2-7b-si-hf'

        model = transformers.LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_name,
            device_map=device_map,
            **common_kwargs
        )
        processor = transformers.AutoProcessor.from_pretrained(model_name)

    return tokenizer, model, processor



def inference_maira2_report(model, processor, image_path, prompt, grounding=False, max_new_tokens=500):
    image = Image.open(image_path).convert('RGB') 
    processed_inputs = processor.format_and_preprocess_reporting_input(
                current_frontal=image,
                current_lateral=None,
                prior_frontal=None,
                indication=None,
                technique=None,
                comparison=None,
                prior_report=None,
                return_tensors="pt",
                get_grounding=False
                ).to(model.device)
    
    output_decoding = model.generate(
                    **processed_inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True
                )
    prompt_length = processed_inputs["input_ids"].shape[-1]
    decoded_text = processor.decode(output_decoding[0][prompt_length:], skip_special_tokens=True)
    decoded_text = decoded_text.lstrip()  # Findings generation completions have a single leading space
    generated_text = processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)

    return generated_text



def inference_maira2_grounding(model, processor, image_path, label, max_new_tokens=500):

    image = Image.open(image_path).convert('RGB') 
    processed_inputs = processor.format_and_preprocess_phrase_grounding_input(
        frontal_image=image,
        phrase=label,
        return_tensors="pt",
    ).to(model.device)
    
    output_decoding = model.generate(
        **processed_inputs, 
        max_new_tokens=max_new_tokens,
        use_cache=True
    )

    prompt_length = processed_inputs["input_ids"].shape[-1]
    decoded_text = processor.decode(output_decoding[0][prompt_length:], skip_special_tokens=True)
    try:
        prediction = processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)

        
        width, height = image.size
        coordinates = [
            list(processor.adjust_box_for_original_image_size(coord, width=width, height=height))
            for coord in prediction[0][1] if coord is not None
        ]
        coordinates_str = ", ".join(str([round(val, 2) for val in box]) for box in coordinates) if coordinates else ""

    except Exception as e:
        print(f"Error occurred: {e}")
        coordinates_str = ""

    return coordinates_str




def inference_radialog(tokenizer, model, image_path, prompt, chat_history=None, max_new_tokens=500):
    """
    Generate a response in a single-turn or multi-turn conversation for the RaDialog model.
    
    This function always returns the updated chat_history and the model's response.
    If `chat_history` is None or empty, it acts as single-turn but still returns the updated chat_history.

    Args:
        tokenizer: The tokenizer corresponding to the RaDialog model.
        model: The RaDialog model.
        image: The PIL image (or similar) used for visual context.
        prompt: The new user prompt for this turn.
        chat_history: A list of (user_msg, assistant_msg) representing the conversation so far.
                      If None or empty, acts as single-turn but will return the new chat_history.
        max_new_tokens: The maximum number of new tokens to generate.

    Returns:
        chat_history (list): The updated chat_history including this turn's user query and assistant response.
        pred (str): The assistant's response for this turn.
    """

    # Initialize chat_history if not provided
    if chat_history is None:
        chat_history = []

    # Check if this is the first turn (single-turn scenario)
    first_turn = (len(chat_history) == 0)

    # Preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = remap_to_uint8(np.array(image))
    image = Image.fromarray(image).convert("L")

    model.config.tokenizer_padding_side = "left"
    conv = conv_vicuna_v1.copy()

    # Rebuild the conversation from history if it's multi-turn
    for human, assistant in chat_history:
        conv.append_message("USER", human)
        conv.append_message("ASSISTANT", assistant)

    # For the very first turn, prepend "<image>. "
    if first_turn:
        user_prompt = "<image>. " + prompt
    else:
        user_prompt = prompt

    # Add the new user message and a placeholder for the assistant's response
    conv.append_message("USER", user_prompt)
    conv.append_message("ASSISTANT", None)

    # Construct the final prompt text
    text_input = conv.get_prompt()

    # Prepare the image tensor
    vis_transforms_biovil = create_chest_xray_transform_for_inference(512, center_crop_size=448)
    image_tensor = vis_transforms_biovil(image).unsqueeze(0)
    image_tensor = image_tensor.to(model.device, dtype=torch.bfloat16)

    # Tokenize input including the image token
    input_ids = tokenizer_image_token(text_input, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

    # Stopping criteria
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    # Generate the response
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode the generated output
    pred = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().replace("</s>", "")

    # Update the conversation: remove the None placeholder and add the actual assistant response
    conv.messages.pop()  # remove the placeholder None
    conv.append_message("ASSISTANT", pred)

    # Update the external chat_history with the new turn
    chat_history.append((prompt, pred))

    return pred, chat_history


class ExpandChannels:
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if data.shape[0] != 1:
            raise ValueError(f"Expected input of shape [1, H, W], found {data.shape}")
        return torch.repeat_interleave(data, 3, dim=0)

def create_chest_xray_transform_for_inference(resize: int, center_crop_size: int) -> Compose:
    transforms = [Resize(resize), CenterCrop(center_crop_size), ToTensor(), ExpandChannels()]
    return Compose(transforms)




def inference_llavamed(model, processor, image_path, prompt, chat_history=None, max_new_tokens=500):
    """
    Unified function for LLaVA-Med inference, supporting both single-turn and multi-turn modes.

    Args:
        model: The LLaVA-Med model.
        processor: The processor for LLaVA-Med (provides apply_chat_template and tokenizer).
        image: The image (PIL or NumPy array) for the first turn, or None for subsequent turns.
        prompt: The user message for this turn.
        chat_history: A list of (user_message, assistant_message) for past turns. If None or empty, single-turn mode is used.
        max_new_tokens: Maximum number of new tokens to generate.

    Returns:
        chat_history: The updated chat_history including this turn's user prompt and the assistant's response.
        response: The assistant's response string for this turn.
    """
    IMAGE_TOKEN_INDEX = -200

    # Initialize chat history if not provided
    if chat_history is None:
        chat_history = []

    # Prepare conversation history
    conversation = []
    for i, (user_text, assistant_text) in enumerate(chat_history):
        if i == 0:
            conversation.append({"role": "user", "content": f"<image>\n{user_text}"})
        else:
            conversation.append({"role": "user", "content": user_text})
        conversation.append({"role": "assistant", "content": assistant_text})

    # Add the current user prompt
    if len(chat_history) == 0:
        # First turn: Add the image token
        user_content = f"<image>\n{prompt}"
    else:
        # Subsequent turns: No image token
        user_content = prompt

    conversation.append({"role": "user", "content": user_content})

    # Apply chat template to prepare the full prompt
    full_prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_special_tokens=True,
        add_generation_prompt=True,
    )

    # Tokenize the prompt and add image token
    input_ids = tokenizer_image_token(
        full_prompt, processor, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).to(model.device)

    inputs = {"inputs": input_ids.unsqueeze(0)}

    # If an image is provided (first turn), preprocess and include it
    image = Image.open(image_path).convert('RGB')
    if image is not None:
        inputs["images"] = (
            model.get_vision_tower()
            .image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
            .to(model.device, torch.float16)
        )

    # Set attention mask
    inputs["attention_mask"] = torch.ones_like(inputs["inputs"])

    # Load generation config
    generation_config = transformers.GenerationConfig.from_pretrained(
        'microsoft/llava-med-v1.5-mistral-7b',
        local_files_only=False, trust_remote_code=True
    )
    generation_config.pad_token_id = processor.pad_token_id

    # Generate output
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            generation_config=generation_config,
            max_new_tokens=max_new_tokens
        )

    # Decode the model's output
    response = processor._tokenizer.decode(output[0].tolist(), skip_special_tokens=True).strip()

    # Update the chat history
    chat_history.append((prompt, response))

    return response, chat_history



    

def inference_llavaov(model, processor, image_path, prompt, chat_history=None, max_new_tokens=1500):
    """
    Generate a response using the LLaVA-OV model in either single-turn or multi-turn mode.

    Args:
        model: The LLaVA-OV model.
        processor: The processor for LLaVA-OV (provides apply_chat_template and tokenization).
        image: The PIL or NumPy image. On the first turn, this will be included with the prompt.
        prompt: The user prompt for this turn.
        chat_history: A list of (user_msg, assistant_msg) tuples representing the conversation so far.
                      If None or empty, single-turn mode is used. Even in single-turn mode, 
                      this function returns chat_history so that you can continue in subsequent turns.
        max_new_tokens: The maximum number of new tokens to generate.

    Returns:
        chat_history (list): The updated chat_history including this turn's (prompt, response).
        response (str): The assistant's response for this turn.
    """

    # If no chat_history provided, initialize an empty one (single-turn scenario)
    if chat_history is None:
        chat_history = []

    # Convert image to the expected shape (C, H, W)

    image = Image.open(image_path).convert('RGB')
    image = asarray(image.convert('RGB')).transpose(2, 0, 1)

    # Prepare the conversation from chat_history
    conversation = []
    num_round = 0
    for user_text, assistant_text in chat_history:
        if num_round==0:
            conversation.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image"},
                    ],
                }
            )
        else:
            conversation.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                    ],
                }
            )
        # Add assistant response from history     
        conversation.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": assistant_text},
                    ],
                }
            )
        num_round += 1

    # Check if this is the first round of conversation
    if len(chat_history) == 0:
        # First turn: Add the user message with the image token
        conversation.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            }
        )
    else:
        # Subsequent turns: Add user message without the image token
        conversation.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        )

    # Generate a response using the model
    full_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Prepare model inputs
    inputs = processor(images=image, text=full_prompt, return_tensors="pt", padding=True).to(
        model.device, torch.float16
    )

    # Generate response
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    full_response = processor.decode(output[0], skip_special_tokens=True)
    response = re.split(r"(user|assistant)", full_response)[-1].strip()

    # Update chat_history
    chat_history.append((prompt, response))

    return response, chat_history



def inference_chexagent(model, tokenizer, image_path, prompt, grounding=False, max_new_tokens=500):
    paths = [image_path]
    query = tokenizer.from_list_format([*[{'image': path} for path in paths], {'text': prompt}])
    conv = [{"from": "system", "value": "You are a helpful assistant."}, {"from": "human", "value": query}]
    input_ids = tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt")
    output = model.generate(
        input_ids.to(model.device), do_sample=False, num_beams=1, temperature=1., top_p=1., use_cache=True,
        max_new_tokens=max_new_tokens
    )[0]
    generated_text = tokenizer.decode(output[input_ids.size(1):-1])

    if grounding:
        pattern = re.compile(r"<\|box\|> \((\d+),(\d+)\),\((\d+),(\d+)\) <\|/box\|>")
        # Find all matches in the text
        matches = pattern.findall(generated_text)
        if not matches:
            return ""
        # Transform the coordinates into the desired format
        result = [
            f"[{int(x1)/100:.2f}, {int(y1)/100:.2f}, {int(x2)/100:.2f}, {int(y2)/100:.2f}]"
            for x1, y1, x2, y2 in matches
        ]
        
        generated_text = ", ".join(result)


    return generated_text
