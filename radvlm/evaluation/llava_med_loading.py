from abc import ABC, abstractmethod
import os
from glob import glob
import re
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    MistralConfig,
    MistralModel,
    MistralForCausalLM,
)
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


MODELS_DIR = '/store/swissai/a02/health_mm_llm_shared/models'


def register_llava_med_hf():
    IGNORE_INDEX = -100
    IMAGE_TOKEN_INDEX = -200
    DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
    DEFAULT_IM_START_TOKEN = "<im_start>"
    DEFAULT_IM_END_TOKEN = "<im_end>"

    def build_vision_tower(vision_tower_cfg, **kwargs):
        vision_tower = getattr(
            vision_tower_cfg,
            "mm_vision_tower",
            getattr(vision_tower_cfg, "vision_tower", None),
        )
        is_absolute_path_exists = os.path.exists(vision_tower)
        if (
            is_absolute_path_exists
            or vision_tower.startswith("openai")
            or vision_tower.startswith("laion")
        ):
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    class IdentityMap(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, *args, **kwargs):
            return x

        @property
        def config(self):
            return {"mm_projector_type": "identity"}

    class SimpleResBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.pre_norm = nn.LayerNorm(channels)

            self.proj = nn.Sequential(
                nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels)
            )

        def forward(self, x):
            x = self.pre_norm(x)
            return x + self.proj(x)

    def build_vision_projector(config, **kwargs):
        projector_type = getattr(config, "mm_projector_type", "linear")

        if projector_type == "linear":
            return nn.Linear(config.mm_hidden_size, config.hidden_size)

        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            return nn.Sequential(*modules)

        if projector_type == "identity":
            return IdentityMap()

        raise ValueError(f"Unknown projector type: {projector_type}")

    class CLIPVisionTower(nn.Module):
        def __init__(self, vision_tower, args, delay_load=False):
            super().__init__()

            self.is_loaded = False

            self.vision_tower_name = vision_tower
            self.select_layer = args.mm_vision_select_layer
            self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

            self.path = 'openai/clip-vit-large-patch14-336'
            self.cfg_only = CLIPVisionConfig.from_pretrained(
                    self.path, local_files_only=False, trust_remote_code=True
                )
            self.load_model()

        def load_model(self):
            # TODO: Once we have internet access, maybe change it s.t. it loads the CLIP for the right image resoltion. For now, we'll just use 336
            self.image_processor = CLIPImageProcessor.from_pretrained(
                self.path, local_files_only=False, trust_remote_code=True
            )
            
            self.vision_tower = CLIPVisionModel.from_pretrained(
                self.path,
                local_files_only=False,
                device_map="cpu",
                trust_remote_code=True,
            )
            self.vision_tower.requires_grad_(False)

            self.is_loaded = True

        def feature_select(self, image_forward_outs):
            image_features = image_forward_outs.hidden_states[self.select_layer]
            if self.select_feature == "patch":
                image_features = image_features[:, 1:]
            elif self.select_feature == "cls_patch":
                image_features = image_features
            else:
                raise ValueError(f"Unexpected select feature: {self.select_feature}")
            return image_features

        @torch.no_grad()
        def forward(self, images):
            if type(images) is list:
                image_features = []
                for image in images:
                    image_forward_out = self.vision_tower(
                        image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                        output_hidden_states=True,
                    )
                    image_feature = self.feature_select(image_forward_out).to(
                        image.dtype
                    )
                    image_features.append(image_feature)
            else:
                image_forward_outs = self.vision_tower(
                    images.to(device=self.device, dtype=self.dtype),
                    output_hidden_states=True,
                )
                image_features = self.feature_select(image_forward_outs).to(
                    images.dtype
                )

            return image_features

        @property
        def dummy_feature(self):
            return torch.zeros(
                1, self.hidden_size, device=self.device, dtype=self.dtype
            )

        @property
        def dtype(self):
            return self.vision_tower.dtype

        @property
        def device(self):
            return self.vision_tower.device

        @property
        def config(self):
            if self.is_loaded:
                return self.vision_tower.config
            else:
                return self.cfg_only

        @property
        def hidden_size(self):
            return self.config.hidden_size

        @property
        def num_patches(self):
            return (self.config.image_size // self.config.patch_size) ** 2

    class LlavaMetaModel:

        def __init__(self, config):
            super(LlavaMetaModel, self).__init__(config)

            if hasattr(config, "mm_vision_tower"):
                self.vision_tower = build_vision_tower(config)
                self.mm_projector = build_vision_projector(config)

        def get_vision_tower(self):
            vision_tower = getattr(self, "vision_tower", None)
            if type(vision_tower) is list:
                vision_tower = vision_tower[0]
            return vision_tower

        def initialize_vision_modules(self, model_args, fsdp=None, embed_tokens=None):
            vision_tower = model_args.vision_tower
            mm_vision_select_layer = model_args.mm_vision_select_layer
            mm_vision_select_feature = model_args.mm_vision_select_feature
            pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

            self.config.mm_vision_tower = vision_tower

            if self.get_vision_tower() is None:
                vision_tower = build_vision_tower(model_args)

                if fsdp is not None and len(fsdp) > 0:
                    self.vision_tower = [vision_tower]
                else:
                    self.vision_tower = vision_tower
            else:
                if fsdp is not None and len(fsdp) > 0:
                    vision_tower = self.vision_tower[0]
                else:
                    vision_tower = self.vision_tower
                vision_tower.load_model()

            self.config.use_mm_proj = True
            self.config.mm_projector_type = getattr(
                model_args, "mm_projector_type", "linear"
            )
            self.config.mm_hidden_size = vision_tower.hidden_size
            self.config.mm_vision_select_layer = mm_vision_select_layer
            self.config.mm_vision_select_feature = mm_vision_select_feature

            # add additional configs for segtok
            self.config.feature_outs = model_args.feature_outs
            self.config.img_size = model_args.img_size
            self.config.vision_backbone = model_args.vision_backbone
            self.config.segtok_posembed = model_args.segtok_posembed

            if getattr(self, "mm_projector", None) is None:
                self.mm_projector = build_vision_projector(self.config)
            else:
                # In case it is frozen by LoRA
                for p in self.mm_projector.parameters():
                    p.requires_grad = True

            # Initialize last layer in mm_projector with weight=0 and bias=mean(embed_tokens)
            if embed_tokens is not None:
                embed_tokens_weight = embed_tokens.weight.data
                self.mm_projector[-1].weight.data.zero_()
                self.mm_projector[-1].bias.data.copy_(embed_tokens_weight.mean(dim=0))

            if pretrain_mm_mlp_adapter is not None:

                def get_w(weights, keyword):
                    return {
                        k.split(keyword + ".")[1]: v
                        for k, v in weights.items()
                        if keyword in k
                    }

                mm_projector_weights = torch.load(
                    pretrain_mm_mlp_adapter, map_location="cpu"
                )
                self.mm_projector.load_state_dict(
                    get_w(mm_projector_weights, "mm_projector")
                )

                # also load additional learnable parameters during feature alignment
                checkpoint_folder = os.path.dirname(pretrain_mm_mlp_adapter)
                ckpts = glob(f"{checkpoint_folder}/checkpoint-*", recursive=False)
                if len(ckpts) > 0:
                    vision_module_weights = torch.load(
                        f"{ckpts[-1]}/mm_projector.bin", map_location="cpu"
                    )
                    model_dict = get_w(vision_module_weights, "vision_tower")
                    print(
                        f"Loading vision module weights from {ckpts[-1]}/mm_projector.bin"
                    )
                    # print keys in model_dict
                    print(f"Loaded keys: {model_dict.keys()}")
                    self.vision_tower.load_state_dict(model_dict, strict=False)

    class LlavaMetaForCausalLM(ABC):

        @abstractmethod
        def get_model(self):
            pass

        def get_vision_tower(self):
            return self.get_model().get_vision_tower()

        def encode_images(self, images):
            image_features = self.get_model().get_vision_tower()(images)
            image_features = self.get_model().mm_projector(image_features)
            return image_features

        def prepare_inputs_labels_for_multimodal(
            self,
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            images,
            image_sizes=None,
        ):
            vision_tower = self.get_vision_tower()
            if vision_tower is None or images is None or input_ids.shape[1] == 1:
                if (
                    past_key_values is not None
                    and vision_tower is not None
                    and images is not None
                    and input_ids.shape[1] == 1
                ):
                    target_shape = past_key_values[-1][-1].shape[-2] + 1
                    attention_mask = torch.cat(
                        (
                            attention_mask,
                            torch.ones(
                                (
                                    attention_mask.shape[0],
                                    target_shape - attention_mask.shape[1],
                                ),
                                dtype=attention_mask.dtype,
                                device=attention_mask.device,
                            ),
                        ),
                        dim=1,
                    )
                    position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
                return (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    None,
                    labels,
                )

            if type(images) is list or images.ndim == 5:
                concat_images = torch.cat([image for image in images], dim=0)
                image_features = self.encode_images(concat_images)
                split_sizes = [image.shape[0] for image in images]
                image_features = torch.split(image_features, split_sizes, dim=0)
                image_features = [
                    x.flatten(0, 1).to(self.device) for x in image_features
                ]
            else:
                image_features = self.encode_images(images).to(self.device)

            # TODO: image start / end is not implemented here to support pretraining.
            if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                self.config, "mm_use_im_start_end", False
            ):
                raise NotImplementedError

            # Let's just add dummy tensors if they do not exist,
            # it is a headache to deal with None all the time.
            # But it is not ideal, and if you have a better idea,
            # please open an issue / submit a PR, thanks.
            _labels = labels
            _position_ids = position_ids
            _attention_mask = attention_mask

            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            else:
                attention_mask = attention_mask.bool()
            if position_ids is None:
                position_ids = torch.arange(
                    0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
                )

            if labels is None:
                labels = torch.full_like(input_ids, IGNORE_INDEX)

            input_ids = [
                cur_input_ids[cur_attention_mask]
                for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
            ]
            labels = [
                cur_labels[cur_attention_mask]
                for cur_labels, cur_attention_mask in zip(labels, attention_mask)
            ]

            new_input_embeds = []
            new_labels = []
            cur_image_idx = 0
            for batch_idx, cur_input_ids in enumerate(input_ids):
                num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
                if num_images == 0:
                    cur_image_features = image_features[cur_image_idx]
                    cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                    cur_input_embeds = torch.cat(
                        [cur_input_embeds_1, cur_image_features[0:0]], dim=0
                    )
                    new_input_embeds.append(cur_input_embeds)
                    new_labels.append(labels[batch_idx])
                    cur_image_idx += 1
                    continue

                image_token_indices = (
                    [-1]
                    + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                    + [cur_input_ids.shape[0]]
                )
                cur_input_ids_noim = []
                cur_labels = labels[batch_idx]
                cur_labels_noim = []
                for i in range(len(image_token_indices) - 1):
                    cur_input_ids_noim.append(
                        cur_input_ids[
                            image_token_indices[i] + 1 : image_token_indices[i + 1]
                        ]
                    )
                    cur_labels_noim.append(
                        cur_labels[
                            image_token_indices[i] + 1 : image_token_indices[i + 1]
                        ]
                    )

                split_sizes = [x.shape[0] for x in cur_labels_noim]
                cur_input_embeds = self.get_model().embed_tokens(
                    torch.cat(cur_input_ids_noim)
                )
                cur_input_embeds_no_im = torch.split(
                    cur_input_embeds, split_sizes, dim=0
                )
                cur_new_input_embeds = []
                cur_new_labels = []

                for i in range(num_images + 1):
                    cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                    cur_new_labels.append(cur_labels_noim[i])
                    if i < num_images:
                        cur_image_features = image_features[cur_image_idx]
                        cur_image_idx += 1
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=cur_labels.device,
                                dtype=cur_labels.dtype,
                            )
                        )

                cur_new_input_embeds = torch.cat(cur_new_input_embeds)
                cur_new_labels = torch.cat(cur_new_labels)

                new_input_embeds.append(cur_new_input_embeds)
                new_labels.append(cur_new_labels)

            # Truncate sequences to max length as image embeddings can make the sequence longer
            tokenizer_model_max_length = getattr(
                self.config, "tokenizer_model_max_length", None
            )
            if tokenizer_model_max_length is not None:
                new_input_embeds = [
                    x[:tokenizer_model_max_length] for x in new_input_embeds
                ]
                new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

            # Combine them
            max_len = max(x.shape[0] for x in new_input_embeds)
            batch_size = len(new_input_embeds)

            new_input_embeds_padded = []
            new_labels_padded = torch.full(
                (batch_size, max_len),
                IGNORE_INDEX,
                dtype=new_labels[0].dtype,
                device=new_labels[0].device,
            )
            attention_mask = torch.zeros(
                (batch_size, max_len),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            position_ids = torch.zeros(
                (batch_size, max_len),
                dtype=position_ids.dtype,
                device=position_ids.device,
            )

            for i, (cur_new_embed, cur_new_labels) in enumerate(
                zip(new_input_embeds, new_labels)
            ):
                cur_len = cur_new_embed.shape[0]
                if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                    new_input_embeds_padded.append(
                        torch.cat(
                            (
                                torch.zeros(
                                    (max_len - cur_len, cur_new_embed.shape[1]),
                                    dtype=cur_new_embed.dtype,
                                    device=cur_new_embed.device,
                                ),
                                cur_new_embed,
                            ),
                            dim=0,
                        )
                    )
                    if cur_len > 0:
                        new_labels_padded[i, -cur_len:] = cur_new_labels
                        attention_mask[i, -cur_len:] = True
                        position_ids[i, -cur_len:] = torch.arange(
                            0,
                            cur_len,
                            dtype=position_ids.dtype,
                            device=position_ids.device,
                        )
                else:
                    new_input_embeds_padded.append(
                        torch.cat(
                            (
                                cur_new_embed,
                                torch.zeros(
                                    (max_len - cur_len, cur_new_embed.shape[1]),
                                    dtype=cur_new_embed.dtype,
                                    device=cur_new_embed.device,
                                ),
                            ),
                            dim=0,
                        )
                    )
                    if cur_len > 0:
                        new_labels_padded[i, :cur_len] = cur_new_labels
                        attention_mask[i, :cur_len] = True
                        position_ids[i, :cur_len] = torch.arange(
                            0,
                            cur_len,
                            dtype=position_ids.dtype,
                            device=position_ids.device,
                        )

            new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

            if _labels is None:
                new_labels = None
            else:
                new_labels = new_labels_padded

            if _attention_mask is None:
                attention_mask = None
            else:
                attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

            if _position_ids is None:
                position_ids = None
            return (
                None,
                position_ids,
                attention_mask,
                past_key_values,
                new_input_embeds,
                new_labels,
            )

        def initialize_vision_tokenizer(self, model_args, tokenizer):
            if model_args.mm_use_im_patch_token:
                tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
                self.resize_token_embeddings(len(tokenizer))

            if model_args.mm_use_im_start_end:
                num_new_tokens = tokenizer.add_tokens(
                    [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
                )
                self.resize_token_embeddings(len(tokenizer))

                if num_new_tokens > 0:
                    input_embeddings = self.get_input_embeddings().weight.data
                    output_embeddings = self.get_output_embeddings().weight.data

                    input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                        dim=0, keepdim=True
                    )
                    output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                        dim=0, keepdim=True
                    )

                    input_embeddings[-num_new_tokens:] = input_embeddings_avg
                    output_embeddings[-num_new_tokens:] = output_embeddings_avg

                if model_args.tune_mm_mlp_adapter:
                    for p in self.get_input_embeddings().parameters():
                        p.requires_grad = True
                    for p in self.get_output_embeddings().parameters():
                        p.requires_grad = False

                if model_args.pretrain_mm_mlp_adapter:
                    mm_projector_weights = torch.load(
                        model_args.pretrain_mm_mlp_adapter, map_location="cpu"
                    )
                    embed_tokens_weight = mm_projector_weights[
                        "model.embed_tokens.weight"
                    ]
                    assert num_new_tokens == 2
                    if input_embeddings.shape == embed_tokens_weight.shape:
                        input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                            -num_new_tokens:
                        ]
                    elif embed_tokens_weight.shape[0] == num_new_tokens:
                        input_embeddings[-num_new_tokens:] = embed_tokens_weight
                    else:
                        raise ValueError(
                            f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                        )
            elif model_args.mm_use_im_patch_token:
                if model_args.tune_mm_mlp_adapter:
                    for p in self.get_input_embeddings().parameters():
                        p.requires_grad = False
                    for p in self.get_output_embeddings().parameters():
                        p.requires_grad = False

    class LlavaMistralConfig(MistralConfig):
        model_type = "llava_mistral"

    class LlavaMistralModel(LlavaMetaModel, MistralModel):
        config_class = LlavaMistralConfig

        def __init__(self, config: MistralConfig):
            super(LlavaMistralModel, self).__init__(config)

    class LlavaMistralForCausalLM(MistralForCausalLM, LlavaMetaForCausalLM):
        config_class = LlavaMistralConfig

        def __init__(self, config):
            super(MistralForCausalLM, self).__init__(config)
            self.model = LlavaMistralModel(config)

            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

            # Initialize weights and apply final processing
            self.post_init()

        def get_model(self):
            return self.model

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            image_sizes: Optional[List[List[int]]] = None,
            return_dict: Optional[bool] = None,
            cache_position=None,
        ) -> Union[Tuple, CausalLMOutputWithPast]:

            if inputs_embeds is None:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    image_sizes,
                )

            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        @torch.no_grad()
        def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            images: Optional[torch.Tensor] = None,
            image_sizes: Optional[torch.Tensor] = None,
            **kwargs,
        ) -> Union[GenerateOutput, torch.LongTensor]:
            position_ids = kwargs.pop("position_ids", None)
            attention_mask = kwargs.pop("attention_mask", None)
            if "inputs_embeds" in kwargs:
                raise NotImplementedError("`inputs_embeds` is not supported")

            if images is not None:
                (inputs, position_ids, attention_mask, _, inputs_embeds, _) = (
                    self.prepare_inputs_labels_for_multimodal(
                        inputs,
                        position_ids,
                        attention_mask,
                        None,
                        None,
                        images,
                        image_sizes=image_sizes,
                    )
                )
            else:
                inputs_embeds = self.get_model().embed_tokens(inputs)

            return super().generate(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
        ):
            images = kwargs.pop("images", None)
            image_sizes = kwargs.pop("image_sizes", None)
            inputs = super().prepare_inputs_for_generation(
                input_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
            if images is not None:
                inputs["images"] = images
            if image_sizes is not None:
                inputs["image_sizes"] = image_sizes
            # inputs["cache_position"] = None
            return inputs

    AutoConfig.register("llava_mistral", LlavaMistralConfig)
    AutoModelForCausalLM.register(LlavaMistralConfig, LlavaMistralForCausalLM)



def tokenizer_image_token(
    prompt, tokenizer, image_token_index=-200, return_tensors=None
):
    prompt_chunks = [
        tokenizer(chunk, add_special_tokens=False).input_ids
        for chunk in prompt.split("<image>")
    ]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids
