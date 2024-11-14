import os
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from ..logging.logging_setup import logger
from .constants import SCHEDULE_TYPES
import re


def generate_lora_tags(names_list, scales_list):
    tags = [
        f"<lora:{os.path.splitext(os.path.basename(str(l_name)))[0]}:{l_scale}>"
        for l_name, l_scale in zip(names_list, scales_list)
        if l_name
    ]
    return " ".join(tags)


def extra_string_metadata(metadata_list):
    parameters_beta = ""
    try:
        if metadata_list[0]:
            parameters_beta += f", VAE: {os.path.splitext(os.path.basename(str(metadata_list[0])))[0]}"
        if metadata_list[1]:
            parameters_beta += f", PAG: {metadata_list[1]}"
        if metadata_list[2]:
            parameters_beta += f", FreeU: {metadata_list[2]}"
        if metadata_list[3]:
            upscaler_param = (
                os.path.splitext(os.path.basename(str(metadata_list[3])))[0]
                if os.path.exists(metadata_list[3])
                else metadata_list[3]
            )
            parameters_beta += f", Hires upscaler: {upscaler_param}"
            parameters_beta += f", Hires upscale: {metadata_list[4]}"
            if metadata_list[5]:
                parameters_beta += f", Hires steps: {metadata_list[5]}"
                parameters_beta += f", Hires denoising strength: {metadata_list[6]}"
        if any(metadata_list[7]):
            lora_params = generate_lora_tags(metadata_list[7], metadata_list[8])
            if lora_params.strip():
                parameters_beta += f", Loras: {lora_params}"
    except Exception as e:
        logger.debug(str(e))
        logger.info("Error generating extra image metadata")

    return parameters_beta


def sanitize_filename(filename):
    # Replace any character that's not alphanumeric, a dash, an underscore, or a period with '_'
    return re.sub(r'[^a-zA-Z0-9_\-.]', '_', filename)


def assemble_filename_pattern(suffix_images, metadata):
    FILENAME_PATTERN = {
        "prompt_section": metadata[0][:15] if metadata[0] else "",
        "neg_prompt_section": metadata[1][:15] if metadata[1] else "",
        "model": os.path.splitext(os.path.basename(str(metadata[2])))[0],
        "vae": os.path.splitext(os.path.basename(str(metadata[3])))[0] if metadata[3] else "",
        "num_steps": metadata[4],
        "guidance_scale": metadata[5],
        "sampler": metadata[6],
        "schedule_type": metadata[11],
        "img_width": metadata[8],
        "img_height": metadata[9],
        "seed": "_STABLEPYSEEDKEY_",
    }

    filename_image = ""
    list_key_words = suffix_images.split(",")
    for key_word in list_key_words:
        if key_word.strip():
            filename_image += "_"
            if key_word in FILENAME_PATTERN:
                filename_image += str(FILENAME_PATTERN[key_word])
            else:
                filename_image += key_word

    return sanitize_filename(filename_image)


def get_string_metadata(metadata_list):
    string_parameters = ""
    try:
        schedule_type = (
            f"Schedule type: {str(metadata_list[11])}, "
            if SCHEDULE_TYPES.get(metadata_list[11], None)
            else ""
        )

        string_parameters = (
            f"{str(metadata_list[0])}\n"
            f"Negative prompt: {str(metadata_list[1])}\n"
            f"Steps: {str(metadata_list[4])}, "
            f"Sampler: {str(metadata_list[6])}, "
            f"{schedule_type}"
            f"CFG scale: {str(metadata_list[5])}, "
            f"Seed: {str(metadata_list[7])}, "
            f"Size: {str(metadata_list[8])}x{str(metadata_list[9])}, "
            f"Model: {os.path.splitext(os.path.basename(str(metadata_list[2])))[0]}, "
            f"Clip skip: {2 if metadata_list[10] else 1}"
        )

        if metadata_list[12]:
            string_parameters += metadata_list[12]

    except Exception as e:
        logger.debug(str(e))
        logger.info("Error generating image metadata")

    return string_parameters


def save_pil_image_with_metadata(
    image,
    folder_path,
    string_parameters,
    suffix="",
):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    existing_files = os.listdir(folder_path)

    # Determine the next available image name
    image_name = f"{str(len(existing_files) + 1).zfill(5)}{suffix}.png"
    image_path = os.path.join(folder_path, image_name)

    try:
        metadata = PngInfo()
        metadata.add_text("parameters", string_parameters)
        image.save(image_path, pnginfo=metadata)
    except Exception as e:
        logger.debug(str(e))
        logger.info("Error saving image with metadata")
        image.save(image_path)

    return image_path


def checkpoint_model_type(checkpoint_path):
    from safetensors.torch import load_file as safe_load

    if not checkpoint_path.endswith(".safetensors"):
        raise ValueError("The checkpoint model is not a SafeTensors file.")

    checkpoint = safe_load(checkpoint_path, device="cpu")
    # checkpoint = torch.load(checkpoint_path, map_location="cpu") # ckpt

    key_name_v2_1 = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
    key_name_sd_xl_base = "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias"
    key_name_sd_xl_refiner = "conditioner.embedders.0.model.transformer.resblocks.9.mlp.c_proj.bias"
    key_name_flux = [
        "double_blocks.0.img_attn.norm.key_norm.scale",
        "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale",
    ]

    # model_type = "v1"
    model_type = "sd1.5"

    if key_name_v2_1 in checkpoint and checkpoint[key_name_v2_1].shape[-1] == 1024:
        # model_type = "v2"
        model_type = "sd2.1"
    elif key_name_sd_xl_base in checkpoint:
        # only base xl has two text embedders
        model_type = "sdxl"
    elif key_name_sd_xl_refiner in checkpoint:
        # only refiner xl has embedder and one text embedders
        model_type = "refiner"
    elif any(key in checkpoint for key in key_name_flux):
        if any(
            g in checkpoint for g in ["guidance_in.in_layer.bias", "model.diffusion_model.guidance_in.in_layer.bias"]
        ):
            model_type = "flux-dev"
        else:
            model_type = "flux-schnell"

    del checkpoint

    return model_type
