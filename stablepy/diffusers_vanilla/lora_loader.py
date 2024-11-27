# =====================================
# LoRA Loaders
# =====================================
import torch
from safetensors.torch import load_file
from collections import defaultdict
from ..logging.logging_setup import logger
import safetensors
import os
import string
import traceback
import logging

VALID_LORA_LAYERS_SDXL = [
    "input_blocks",
    "middle_block",
    "output_blocks",
    "text_model",
    ".down_blocks",
    ".mid_block",
    ".up_blocks",
    # "text_projection",  # text encoder 2 layer
    # "conv_in",  # unet extra layers
    # "time_proj",
    # "time_embedding",
    # "time_embedding.linear_1",
    # "time_embedding.act",
    # "time_embedding.linear_2",
    # "add_time_proj",
    # "add_embedding",
    # "add_embedding.linear_1",
    # "add_embedding.linear_2",
    # "conv_norm_out",
    # "conv_out"
]


def load_lora_weights(pipeline, checkpoint_path, multiplier, device, dtype):
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    # load LoRA weight from .safetensors
    if isinstance(checkpoint_path, str):
        checkpoint_path = [checkpoint_path]
    for ckptpath in checkpoint_path:
        state_dict = load_file(ckptpath, device=device)

        updates = defaultdict(dict)
        for key, value in state_dict.items():
            # it is suggested to print out the key, it usually will be something like below
            # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

            layer, elem = key.split(".", 1)
            updates[layer][elem] = value

        # directly update weight in diffusers model
        for layer, elems in updates.items():
            if "text" in layer:
                layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split(
                    "_"
                )
                curr_layer = pipeline.text_encoder
            else:
                layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
                curr_layer = pipeline.unet

            # find the target layer
            temp_name = layer_infos.pop(0)
            while len(layer_infos) > -1:
                try:
                    curr_layer = curr_layer.__getattr__(temp_name)
                    if len(layer_infos) > 0:
                        temp_name = layer_infos.pop(0)
                    elif len(layer_infos) == 0:
                        break
                except Exception:
                    if len(temp_name) > 0:
                        temp_name += "_" + layer_infos.pop(0)
                    else:
                        temp_name = layer_infos.pop(0)

            # get elements for this layer
            weight_up = elems["lora_up.weight"].to(dtype)
            weight_down = elems["lora_down.weight"].to(dtype)
            alpha = elems["alpha"]
            if alpha:
                alpha = alpha.item() / weight_up.shape[1]
            else:
                alpha = 1.0

            # update weight
            if len(weight_up.shape) == 4:
                curr_layer.weight.data += (
                    multiplier
                    * alpha
                    * torch.mm(
                        weight_up.squeeze(3).squeeze(2),
                        weight_down.squeeze(3).squeeze(2),
                    )
                    .unsqueeze(2)
                    .unsqueeze(3)
                )
            else:
                curr_layer.weight.data += (
                    multiplier * alpha * torch.mm(weight_up, weight_down)
                )

    logger.debug(f"Config LoRA: multiplier {multiplier} | alpha {alpha}")

    return pipeline


def validate_lora_layers(lora_path):
    state_dict = safetensors.torch.load_file(lora_path, device="cpu")
    state_dict = {
        k: w for k, w in state_dict.items()
        if any(ly in k for ly in VALID_LORA_LAYERS_SDXL)
    }

    return state_dict


def lora_mix_load(pipe, lora_path, alpha_scale=1.0, device="cuda", dtype=torch.float16):
    if hasattr(pipe, "text_encoder_2"):
        # sdxl lora
        try:
            pipe.load_lora_weights(lora_path)
            pipe.fuse_lora(lora_scale=alpha_scale)
            pipe.unload_lora_weights()
        except Exception as e:
            pipe.unload_lora_weights()
            if "size mismatch for" in str(e) or not os.path.exists(lora_path):
                raise e

            logger.debug(str(e))

            state_dict = validate_lora_layers(lora_path)

            if not state_dict:
                raise ValueError("No valid lora layers were found.")

            try:
                pipe.load_lora_weights(state_dict)
                pipe.fuse_lora(lora_scale=alpha_scale)
                pipe.unload_lora_weights()
            except Exception as e:
                pipe.unload_lora_weights()
                raise e
    else:
        # sd lora
        try:
            pipe = load_lora_weights(
                pipe, [lora_path], alpha_scale, device=device, dtype=dtype
            )
        except Exception as e:
            logger.debug(f"{str(e)} \nDiffusers loader>>")
            try:
                pipe.load_lora_weights(lora_path)
                pipe.fuse_lora(lora_scale=alpha_scale)
                pipe.unload_lora_weights()
            except Exception as e:
                pipe.unload_lora_weights()
                raise e

    return pipe


def load_no_fused_lora(pipe, num_loras, current_lora_list, current_lora_scale_list):

    lora_status = [None] * num_loras

    logger.debug("Unloading and reloading LoRA weights on the fly")
    pipe.unload_lora_weights()

    active_adapters = []
    active_adapters_scales = []
    number_to_value = {i: letter for i, letter in enumerate(string.ascii_uppercase[:num_loras])}

    for i, (lora, scale) in enumerate(zip(current_lora_list, current_lora_scale_list)):
        if lora:
            try:
                adapter_name = number_to_value[i]
                pipe.load_lora_weights(lora, adapter_name=adapter_name)
                active_adapters.append(adapter_name)
                active_adapters_scales.append(scale)
                lora_status[i] = True
                logger.info(f"Loaded LoRA on the fly: {lora}")
            except Exception as e:
                lora_status[i] = False
                if lora in pipe.get_active_adapters():
                    pipe.delete_adapters(lora)

                if "size mismatch for" in str(e) or not os.path.exists(lora):
                    if logger.isEnabledFor(logging.DEBUG):
                        traceback.print_exc()
                    raise RuntimeError(f"ERROR > LoRA not compatible: {lora}")

                state_dict = validate_lora_layers(lora)

                if not state_dict:
                    logger.debug("No valid LoRA layers were found.")
                    if logger.isEnabledFor(logging.DEBUG):
                        traceback.print_exc()
                    raise RuntimeError(f"ERROR > LoRA not compatible: {lora}")

                try:
                    pipe.load_lora_weights(state_dict, adapter_name=adapter_name)
                    active_adapters.append(adapter_name)
                    active_adapters_scales.append(scale)
                    logger.info(f"Loaded LoRA on the fly: {lora}")
                    lora_status[i] = True
                except Exception:
                    if lora in pipe.get_active_adapters():
                        pipe.delete_adapters(lora)
                    if logger.isEnabledFor(logging.DEBUG):
                        traceback.print_exc()
                    raise RuntimeError(f"ERROR > LoRA not compatible: {lora}")

    if active_adapters:
        pipe.set_adapters(active_adapters, adapter_weights=active_adapters_scales)

    return lora_status  # A wrongly loaded LoRA can cause issues.
