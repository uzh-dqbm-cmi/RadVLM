# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert LLaVa-Onevision checkpoints from the original repository.

URL: https://github.com/LLaVA-VL/LLaVA-NeXT/tree/main

"""

import argparse
import gc
import glob
import json
from pathlib import Path

import os

import requests
import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from safetensors import safe_open

from transformers import (
    AddedToken,
    AutoConfig,
    AutoTokenizer,
    LlavaOnevisionConfig,
    LlavaOnevisionForConditionalGeneration,
    LlavaOnevisionImageProcessor,
    LlavaOnevisionProcessor,
    LlavaOnevisionVideoProcessor,
    SiglipVisionConfig,
)


KEYS_TO_MODIFY_MAPPING = {
    "model.vision_tower.": "",
    "model.mm_projector": "multi_modal_projector",
    "model": "model.model",
    "vision_model.model": "vision_model",
    "lm_head": "language_model.lm_head",
    "model.model": "language_model.model",
    "multi_modal_projector.0": "multi_modal_projector.linear_1",
    "multi_modal_projector.2": "multi_modal_projector.linear_2",
    "language_model.model.image_newline": "image_newline",
}

CHECKPOINTS_DIR = '/capstor/scratch/cscs/ndeperr/checkpoints/'

chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n'}}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all video then #}{% for content in message['content'] | selectattr('type', 'equalto', 'video') %}{{ '<video>\n' }}{% endfor %}{# Render all text next #}{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] }}{% endfor %}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{% generation %}{{ content['text'] }}{% endgeneration %}{% endfor %}{% endif %}{{'<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


def load_original_state_dict(model_path):
    original_state_dict = {}
    for path in glob.glob(f"{model_path}/*"):
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    original_state_dict[key] = f.get_tensor(key)

    # tied weights so lm.head is not saved. Let's clone to load state dict
    if "lm_head.weight" not in original_state_dict:
        original_state_dict["lm_head.weight"] = original_state_dict["model.embed_tokens.weight"].clone()

    return original_state_dict

def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith(".inv_freq"):
            continue
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        new_state_dict[key] = value.to(torch.float16)
    return new_state_dict


def load_image():
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


def convert_llava_to_hf(model_id, model_path, pytorch_dump_folder_path, push_to_hub=False):
    # load original config
    filepath = f"{model_path}/config.json"
    # read json
    with open(filepath) as f:
        data = json.load(f)
        print(data)

    if model_id in ["lmms-lab/llava-onevision-qwen2-0.5b-ov", "lmms-lab/llava-onevision-qwen2-0.5b-si"]:
        text_model_id = "Qwen/Qwen2-0.5B-Instruct"
    elif model_id in ["lmms-lab/llava-onevision-qwen2-7b-ov", "lmms-lab/llava-onevision-qwen2-7b-si"]:
        text_model_id = "Qwen/Qwen2-7B-Instruct"
    elif model_id in ["lmms-lab/llava-onevision-qwen2-72b-ov", "lmms-lab/llava-onevision-qwen2-72b-si"]:
        text_model_id = "Qwen/Qwen2-72B-Instruct"

    vision_model_id = data["mm_vision_tower"]
    torch.set_default_dtype(torch.float16)
    text_config = AutoConfig.from_pretrained(text_model_id)

    tokenizer = AutoTokenizer.from_pretrained(text_model_id, use_fast=True)
    tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)
    tokenizer.add_tokens(AddedToken("<video>", special=True, normalized=False), special_tokens=True)

    image_processor = LlavaOnevisionImageProcessor.from_pretrained(vision_model_id)
    video_processor = LlavaOnevisionVideoProcessor.from_pretrained(vision_model_id)
    processor = LlavaOnevisionProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        video_processor=video_processor,
        num_image_tokens=729,
        vision_feature_select_strategy="full",
        chat_template=chat_template,
    )

    vision_config = SiglipVisionConfig(
        hidden_size=1152,
        image_size=384,
        intermediate_size=4304,
        num_attention_heads=16,
        num_hidden_layers=26,  # drop the last layer
        patch_size=14,
        vision_use_head=False,  # no head
    ).to_dict()

    config = LlavaOnevisionConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config,
        use_image_newline_parameter=True,
    )

    with init_empty_weights():
        model = LlavaOnevisionForConditionalGeneration(config)

    # load original state dict
    state_dict = load_original_state_dict(model_path)
    state_dict = convert_state_dict_to_hf(state_dict)
    model.load_state_dict(state_dict, assign=True)
    model.eval()

    pre_expansion_embeddings = model.language_model.model.embed_tokens.weight.data
    mu = torch.mean(pre_expansion_embeddings, dim=0).float()
    n = pre_expansion_embeddings.size()[0]
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

    # We add an image token so we resize the model
    # Pad to 64 for performance reasons
    # Qwen-based models have extra unused space in the vocab size already, so no need to resize
    pad_shape = 64
    vocab_size = config.text_config.vocab_size
    num_tokens = vocab_size + 2
    model.resize_token_embeddings(num_tokens, pad_to_multiple_of=pad_shape)
    model.language_model.model.embed_tokens.weight.data[vocab_size:] = torch.stack(
        tuple(
            (dist.sample() for _ in range(model.language_model.model.embed_tokens.weight.data[vocab_size:].shape[0]))
        ),
        dim=0,
    )
    model.language_model.lm_head.weight.data[vocab_size:] = torch.stack(
        tuple((dist.sample() for _ in range(model.language_model.lm_head.weight.data[vocab_size:].shape[0]))),
        dim=0,
    )

    print(f"Saving model and processor for {model_id} to {pytorch_dump_folder_path}")
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)
    processor.save_pretrained(pytorch_dump_folder_path)

    # Make space so we can load the model properly now.
    del state_dict
    gc.collect()

    # Load everything back for inference tests in float32 because prev script was written as that
    # Though it's mostly loaded in fp16 as original weights are in fp16
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        pytorch_dump_folder_path, torch_dtype="float16", device_map="auto"
    )
    processor = LlavaOnevisionProcessor.from_pretrained(pytorch_dump_folder_path)
    device = model.device

    # prepare inputs
    image = load_image()
    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nWhat is shown in this image?<|im_end|>\n<|im_start|>assistant\n"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(torch.float16)

    # verify inputs
    filepath = hf_hub_download(
        repo_id="RaushanTurganbay/test-image", filename="llava_onevision_pixel_values.pt", repo_type="dataset"
    )
    original_pixel_values = torch.load(filepath, map_location="cpu")
    assert torch.allclose(original_pixel_values, inputs.pixel_values.half())

    image_sizes = torch.tensor([[899, 1024]])
    assert image_sizes[0].tolist() == inputs.image_sizes[0].tolist()

    # verify single forward pass
    print("Single forward pass")
    with torch.inference_mode():
        inputs = inputs.to(device)
        outputs = model(**inputs)
        print("Shape of logits:", outputs.logits.shape)
        print("First values of logits:", outputs.logits[0, :3, :3])

   
    # verify generation
    output_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        use_cache=True,
    )

    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    print("Generated text:", repr(generated_text))

    if model_id == "lmms-lab/llava-onevision-qwen2-0.5b-si":
        expected_text = "system\nYou are a helpful assistant.\nuser\n\nWhat is shown in this image?\nassistant\nThe image is a radar chart that shows the performance of different algorithms or models in a specific domain, such as image classification or natural language processing. The chart is color-coded to represent different algorithms, with each color corresponding to a specific algorithm. The algorithms are labeled as BLIP-2, InstructBLIP, Owen-VL-Chat, and LLaVA-1.5. The chart also includes a legend at the bottom that explains the color coding and the algorithms represented."
    elif model_id == "lmms-lab/llava-onevision-qwen2-0.5b-ov":
        expected_text = "system\nYou are a helpful assistant.\nuser\n\nWhat is shown in this image?\nassistant\nThe image is a radar chart that compares the performance of different models in a specific task, likely related to natural language processing or machine learning. The chart is divided into different categories, each represented by a different color and labeled with the name of the model or technique used. The models are evaluated based on their performance metrics, such as BLEU-2, InstructBLIP, Qwen-VL-Chat, and LLaVA-1.5. The radar chart helps to visualize the relative"
    elif model_id == "lmms-lab/llava-onevision-qwen2-7b-si":
        expected_text = "system\nYou are a helpful assistant.\nuser\n\nWhat is shown in this image?\nassistant\nThis image is a radar chart that compares the performance of different models on various metrics. The models being compared are BLIP-2, InstructBLIP, and Qwen-VL-Chat. The metrics being compared are VQA, QA, GQA, VQA-av2, and VQA-av2. The chart shows that BLIP-2 performs the best on all metrics, followed by InstructBLIP and Qwen-VL-Chat."
    elif model_id == "lmms-lab/llava-onevision-qwen2-7b-ov":
        expected_text = "system\nYou are a helpful assistant.\nuser\n\nWhat is shown in this image?\nassistant\nThe image shows a radar chart, also known as a spider chart or a star chart, which is used to compare multiple quantitative variables. Each axis represents a different variable, and the chart is filled with data points that represent the performance or values of different entities across these variables.\n\nIn this particular radar chart, the variables are represented on the axes, and the performance of different models or systems is shown by the lines connecting the data points. The models or systems are labeled along the bottom of the chart,"
    elif model_id == "lmms-lab/llava-onevision-qwen2-72b-si":
        expected_text = "system\nYou are a helpful assistant.\nuser\n\nWhat is shown in this image?\nassistant\nThe image shows a radar chart, which is a graphical method of displaying multivariate data in the form of a two-dimensional chart of three or more quantitative variables represented on axes starting from the same point. The chart is used to compare the performance of different models or systems across various benchmarks or metrics.\n\nIn this specific radar chart, there are multiple axes, each representing a different benchmark or metric, such as VQA2, GQA, TextVQA, and others. The chart includes several colored lines"
    elif model_id == "lmms-lab/llava-onevision-qwen2-72b-ov":
        expected_text = "system\nYou are a helpful assistant.\nuser\n\nWhat is shown in this image?\nassistant\nThe image is a radar chart comparing the performance of different models on various multimodal benchmarks. The models compared are BLIP-2, InstructBLIP, POPE, QWen-VL-Chat, and LLava-1.5. The benchmarks include VQAv2, GQA, TextVQA, SQA-IMG, VizWiz, MM-IMDb, MM-VQA, MM-IMDb-CN, MM-IMDb-EN, MM-"
    else:
        raise ValueError(f"Model {model_id} not supported")

    # assert generated_text == expected_text
    print("Generated text is ok!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        help="Hub location of the model to convert",
        default="lmms-lab/llava-onevision-qwen2-7b-si",
        choices=[
            "lmms-lab/llava-onevision-qwen2-0.5b-ov",
            "lmms-lab/llava-onevision-qwen2-0.5b-si",
            "lmms-lab/llava-onevision-qwen2-7b-si",
            "lmms-lab/llava-onevision-qwen2-7b-ov",
            "lmms-lab/llava-onevision-qwen2-72b-si",
            "lmms-lab/llava-onevision-qwen2-72b-ov",
        ],
        required=False,
    )

    parser.add_argument(
    "--model_path", default=None, type=str, help="Path to local model."
    )

    args = parser.parse_args()


    model_full_path = os.path.join(CHECKPOINTS_DIR, args.model_path)
    model_path_hf = model_full_path + "_hf"

    if not os.path.exists(model_path_hf):
        convert_llava_to_hf(args.model_id, model_full_path, model_path_hf)
    else:
        print("Conversion already done!")


