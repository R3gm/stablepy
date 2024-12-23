import os
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from ..logging.logging_setup import logger
from .constants import SCHEDULE_TYPES
import re
import torch
import numpy as np
from diffusers.utils.import_utils import is_accelerate_available
from diffusers import ControlNetModel
from accelerate import init_empty_weights
from huggingface_hub import model_info as model_info_data
from diffusers.pipelines.pipeline_loading_utils import variant_compatible_siblings
import hashlib
from collections import OrderedDict
import gc
import inspect
import random
from huggingface_hub import hf_hub_download

MAIN_REPO_FLUX = [
    "camenduru/FLUX.1-dev-diffusers",
    "black-forest-labs/FLUX.1-dev",
    "multimodalart/FLUX.1-dev2pro-full",
]


def get_flux_components_info():
    for repo_ in MAIN_REPO_FLUX:
        try:
            config = hf_hub_download(
                repo_id=repo_,
                filename="model_index.json",
            )
            return repo_, config
        except Exception as e:
            logger.debug(e)
    raise RuntimeError(f"Can't get components info of the Flux repos: {MAIN_REPO_FLUX}")


class CURRENT_TASK_PARAMS:
    def __init__(self, **kwargs):
        if kwargs:
            self.set_params(**kwargs)

    def set_params(self, **kwargs):
        for name_param, value_param in kwargs.items():
            setattr(self, name_param, value_param)


def get_seeds(pp, task_name, device):
    # number seed
    if pp.seed == -1:
        seeds = [random.randint(0, 2147483647)]
    else:
        seeds = [pp.seed]

    seeds = [seeds[0] + i for i in range(pp.num_images)]

    generators = []  # List to store all the generators
    for calculate_seed in seeds:
        if pp.generator_in_cpu or device.type == "cpu":
            generator = torch.Generator().manual_seed(calculate_seed)
        else:
            try:
                generator = torch.Generator("cuda").manual_seed(calculate_seed)
            except Exception as e:
                logger.debug(str(e))
                logger.warning("Generator in CPU")
                generator = torch.Generator().manual_seed(calculate_seed)

        generators.append(generator)

    # fix img2img bug need concat tensor prompts with generator same number (only in batch inference)
    generators = generators if task_name != "img2img" else generators[0]  # no list
    seeds = seeds if task_name != "img2img" else [seeds[0]] * pp.num_images
    return seeds, generators


def generate_lora_tags(names_list, scales_list):
    tags = [
        f"<lora:"
        f"{os.path.splitext(os.path.basename(str(l_name)))[0] if os.path.exists(l_name) else os.path.basename(str(l_name))}:"
        f"{l_scale}>"
        for l_name, l_scale in zip(names_list, scales_list)
        if l_name
    ]
    return " ".join(tags)


def extra_string_metadata(metadata_list):
    parameters_beta = ""
    try:
        if metadata_list[0]:
            parameters_beta += f", VAE: {os.path.splitext(os.path.basename(str(metadata_list[0])))[0] if os.path.exists(metadata_list[0]) else os.path.basename(str(metadata_list[0]))}"
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
        "model": os.path.splitext(os.path.basename(str(metadata[2])))[0] if os.path.exists(metadata[2]) else os.path.basename(str(metadata[2])),
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
            f"Model: {os.path.splitext(os.path.basename(str(metadata_list[2])))[0] if os.path.exists(metadata_list[2]) else os.path.basename(str(metadata_list[2]))}, "
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
    scheduler_config = {}
    sampling_prediction_type = None

    has_baked_vae = any(k.startswith("first_stage_model.") for k in checkpoint.keys())

    if key_name_v2_1 in checkpoint and checkpoint[key_name_v2_1].shape[-1] == 1024:
        # model_type = "v2"
        model_type = "sd2.1"
    elif key_name_sd_xl_base in checkpoint:
        # only base xl has two text embedders
        model_type = "sdxl"

        if 'edm_mean' in checkpoint and 'edm_std' in checkpoint:  # Playground V2.5
            scheduler_config["sigma_data"] = 0.5
            scheduler_config["sigma_max"] = 80.0
            scheduler_config["sigma_min"] = 0.002
            sampling_prediction_type = "EDM"
        elif "edm_vpred.sigma_max" in checkpoint:
            scheduler_config["sigma_max"] = float(checkpoint["edm_vpred.sigma_max"].item())
            if "edm_vpred.sigma_min" in checkpoint:
                scheduler_config["sigma_min"] = float(checkpoint["edm_vpred.sigma_min"].item())
            sampling_prediction_type = "V_PREDICTION_EDM"
        elif "v_pred" in checkpoint:
            scheduler_config["prediction_type"] = "v_prediction"
            scheduler_config["rescale_betas_zero_snr"] = True
            sampling_prediction_type = "V_PREDICTION"
        else:
            sampling_prediction_type = "EPS"

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

    return model_type, sampling_prediction_type, scheduler_config, has_baked_vae


def load_cn_diffusers(model_path, base_config, torch_dtype):
    from diffusers.loaders.single_file_utils import convert_controlnet_checkpoint
    from diffusers.models.model_loading_utils import load_state_dict, load_model_dict_into_meta

    state_dict = load_state_dict(model_path)

    while "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if "time_embedding.linear_1.weight" not in state_dict:
        # https://github.com/huggingface/diffusers/blob/cd6ca9df2987c000b28e13b19bd4eec3ef3c914b/src/diffusers/loaders/single_file_model.py#L50
        state_dict = convert_controlnet_checkpoint(state_dict, base_config)

    if not state_dict:
        raise ValueError(
            f"Failed to load {model_path}. Weights for this component appear to be missing in the checkpoint."
        )

    config = ControlNetModel.load_config(base_config)

    with init_empty_weights():
        controlnet_model = ControlNetModel.from_config(config)

    if is_accelerate_available():
        unexpected_keys = load_model_dict_into_meta(controlnet_model, state_dict, dtype=torch_dtype)

    else:
        _, unexpected_keys = controlnet_model.load_state_dict(state_dict, strict=False)

    controlnet_model.to(torch_dtype)
    controlnet_model.eval()

    return controlnet_model


def check_variant_file(model_id, variant):
    try:
        info = model_info_data(
            model_id,
            timeout=5.0,
        )

        filenames = {sibling.rfilename for sibling in info.siblings}
        _, variant_filenames = variant_compatible_siblings(
            filenames, variant=variant
        )
    except Exception as e:
        logger.debug(str(e))
        variant_filenames = None

    return variant if variant_filenames else None


def process_prompts_valid(specific_prompt, specific_negative_prompt, prompt, negative_prompt):
    specific_prompt_empty = (specific_prompt in [None, ""])
    specific_negative_prompt_empty = (specific_negative_prompt in [None, ""])

    prompt_valid = prompt if specific_prompt_empty else specific_prompt
    negative_prompt_valid = negative_prompt if specific_negative_prompt_empty else specific_negative_prompt

    return specific_prompt_empty, specific_negative_prompt_empty, prompt_valid, negative_prompt_valid


def convert_image_to_numpy_array(image, gui_active=False):
    if isinstance(image, str):
        # If the input is a string (file path), open it as an image
        image_pil = Image.open(image)
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        numpy_array = np.array(image_pil, dtype=np.uint8)
    elif isinstance(image, Image.Image):
        # If the input is already a PIL Image, convert it to a NumPy array
        if image.mode != 'RGB':
            image = image.convert('RGB')
        numpy_array = np.array(image, dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        # If the input is a NumPy array, ensure it's np.uint8
        numpy_array = image.astype(np.uint8)
    else:
        if gui_active:
            logger.info("Not found image")
            return None
        else:
            raise ValueError(
                "Unsupported image type or not control image found; Bug report to https://github.com/R3gm/stablepy or https://github.com/R3gm/SD_diffusers_interactive"
            )

    # Extract the RGB channels
    try:
        array_rgb = numpy_array[:, :, :3]
    except Exception as e:
        logger.error(str(e))
        logger.error("Unsupported image type")
        raise ValueError(
            "Unsupported image type; Bug report to https://github.com/R3gm/stablepy or https://github.com/R3gm/SD_diffusers_interactive"
        )

    return array_rgb


def latents_to_rgb(latents, latent_resize, vae_decoding, pipe):
    weights = (
        (60, -60, 25, -70),
        (60, -5, 15, -50),
        (60, 10, -5, -35)
    )
    if vae_decoding:
        with torch.no_grad():
            latents = [tl.unsqueeze(0) for tl in torch.unbind(latents, dim=0)][0]
            image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]

        resized_image = pipe.image_processor.postprocess(image, output_type="pil")
    else:
        weights_tensor = torch.tensor(weights, dtype=latents.dtype, device=latents.device).T

        biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype, device=latents.device)
        rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.view(-1, 1, 1)
        image_array = rgb_tensor.clamp(0, 255)[0].byte().cpu().numpy()
        image_array = image_array.transpose(1, 2, 0)  # Change the order of dimensions

        pil_image = Image.fromarray(image_array)

        resized_image = pil_image.resize((pil_image.size[0] * latent_resize, pil_image.size[1] * latent_resize), Image.LANCZOS)  # Resize 128x128 * ...

    return resized_image


DEPRECATED_PARAMS = {
    "esrgan_tile": "upscaler_tile_size",
    "esrgan_tile_overlap": "upscaler_tile_overlap",
}


def validate_and_update_params(cls, kwargs, config):
    """
    Validates kwargs against the parameters of a given class's `__call__` method
    and updates the provided configuration dictionary.

    Args:
        cls: The class whose `__call__` method parameters are used for validation.
        kwargs (dict): The keyword arguments to validate.
        config (dict): The dictionary to update with valid parameters.
    """
    if kwargs:
        # logger.debug(kwargs)
        valid_params = inspect.signature(cls.__call__).parameters.keys()
        for name_param, value_param in kwargs.items():
            if name_param in valid_params:
                config.update({name_param: value_param})
                logger.debug(f"Parameter added: '{name_param}': {value_param}.")
            elif name_param in DEPRECATED_PARAMS:
                logger.warning(
                    f"The parameter '{name_param}' is deprecated. Please "
                    f"use '{DEPRECATED_PARAMS[name_param]}' instead."
                )
            else:
                logger.error(
                    f"The pipeline '{cls.__name__}' had an invalid parameter"
                    f" removed: '{name_param}'.")


def cachebox(max_cache_size=None, hash_func=hashlib.md5):
    """
    A decorator to cache the results of a function call, similar to `functools.lru_cache`.
    Args:
        max_cache_size (int, optional): The maximum number of items to store in the cache. If None, the cache size is unlimited.
        hash_func (callable, optional): A hash function to compute the cache keys. Defaults to `hashlib.md5`.
    Returns:
        callable: A decorator that caches the results of the decorated function.
    The decorated function will have an additional `memory` attribute, which is an `OrderedDict` storing the cached results.
    Example:
        @cachebox(max_cache_size=100)
        def expensive_function(x, y):
            # Expensive computation here
            return x + y
        result = expensive_function(1, 2)
    """

    def decorator(func):
        func.memory = OrderedDict()

        def _compute_hash(*args, **kwargs):
            """Compute a hash key from the given arguments."""
            key_string = "|".join(map(str, args)) + "|" + "|".join(f"{k}:{v}" for k, v in kwargs.items())
            return hash_func(key_string.encode()).hexdigest()

        def wrapper(*args, **kwargs):
            # Check if the function is a method (bound to a class instance)
            if len(args) > 0 and hasattr(args[0], "__class__"):
                # Exclude the first argument (self) from the key
                key = _compute_hash(*args[1:], **kwargs)
            else:
                # Use all arguments if it's not a method
                key = _compute_hash(*args, **kwargs)

            if key in func.memory:
                logger.debug(f"Fetching result from memory for hash key: {key}")
                # Move the key to the end to mark it as recently used
                func.memory.move_to_end(key)
                return func.memory[key]
            else:
                logger.debug(f"Computing result for hash: {key}")
                result = func(*args, **kwargs)
                func.memory[key] = result
                # Check if the cache size exceeds the limit
                if max_cache_size is not None and len(func.memory) > max_cache_size:
                    # Remove the oldest item
                    removed_key, removed_value = func.memory.popitem(last=False)
                    logger.debug(f"Evicting oldest cache entry: {removed_key}")
                return result

        # Ensure the memory attribute is accessible on the decorated function
        wrapper.memory = func.memory
        return wrapper
    return decorator


def release_resources():
    torch.cuda.empty_cache()
    gc.collect()
