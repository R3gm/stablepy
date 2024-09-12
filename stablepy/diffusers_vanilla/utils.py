import os
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from ..logging.logging_setup import logger


def save_pil_image_with_metadata(image, folder_path, metadata_list):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    existing_files = os.listdir(folder_path)

    # Determine the next available image name
    image_name = f"image{str(len(existing_files) + 1).zfill(3)}.png"
    image_path = os.path.join(folder_path, image_name)

    try:
        # metadata
        metadata = PngInfo()
        string_parameters = f"{str(metadata_list[0])}, Negative prompt: {str(metadata_list[1])} Steps: {str(metadata_list[4])}, Sampler: {str(metadata_list[6])}, CFG scale: {str(metadata_list[5])}, Seed: {str(metadata_list[7])}, Size: {str(metadata_list[8])}x{str(metadata_list[9])}, Model: {os.path.splitext(str(metadata_list[2]).split('/')[-1])[0]}, Clip skip: {2 if metadata_list[10] else 1},"
        metadata.add_text("parameters", string_parameters)
        # metadata.add_text("Prompt", str(metadata_list[0]))
        # metadata.add_text("Negative prompt", str(metadata_list[1]))
        # metadata.add_text("Model", str(metadata_list[2]))
        # metadata.add_text("VAE", str(metadata_list[3]))
        # metadata.add_text("Steps", str(metadata_list[4]))
        # metadata.add_text("CFG", str(metadata_list[5]))
        # metadata.add_text("Scheduler", str(metadata_list[6]))
        # metadata.add_text("Seed", str(metadata_list[7]))

        image.save(image_path, pnginfo=metadata)
    except Exception:
        logger.info("Saving image without metadata")
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

    del checkpoint

    return model_type
    