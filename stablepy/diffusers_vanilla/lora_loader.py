# =====================================
# LoRA Loaders
# =====================================
import torch
from safetensors.torch import load_file
from collections import defaultdict
from ..logging.logging_setup import logger

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


def lora_mix_load(pipe, lora_path, alpha_scale=1.0, device="cuda", dtype=torch.float16):
    if hasattr(pipe, "text_encoder_2"):
        # sdxl lora
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora(lora_scale=alpha_scale)
    else:
        # sd lora
        try:
            pipe = load_lora_weights(
                pipe, [lora_path], alpha_scale, device=device, dtype=dtype
            )
        except Exception as e:
            logger.debug(f"{str(e)} \nDiffusers loader>>")
            pipe.load_lora_weights(lora_path)
            pipe.fuse_lora(lora_scale=alpha_scale)

    return pipe
