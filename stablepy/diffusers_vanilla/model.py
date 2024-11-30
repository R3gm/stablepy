import gc
import time
import numpy as np
import PIL.Image
from diffusers import (
    ControlNetModel,
    DiffusionPipeline,
    StableDiffusionPipeline,
    AutoencoderKL,
    T2IAdapter,
    StableDiffusionXLPipeline,
    AutoPipelineForImage2Image,
)
from huggingface_hub import hf_hub_download
import torch
import random
import threading
import json
from controlnet_aux import (
    CannyDetector,
    ContentShuffleDetector,
    HEDdetector,
    LineartAnimeDetector,
    LineartDetector,
    MidasDetector,
    MLSDdetector,
    NormalBaeDetector,
    OpenposeDetector,
    PidiNetDetector,
)
from transformers import pipeline
from controlnet_aux.util import HWC3, ade_palette
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
import cv2
from diffusers import (
    DDIMScheduler,
)
from .constants import (
    CLASS_DIFFUSERS_TASK,
    CLASS_PAG_DIFFUSERS_TASK,
    CONTROLNET_MODEL_IDS,
    VALID_TASKS,
    SD15_TASKS,
    SDXL_TASKS,
    T2I_PREPROCESSOR_NAME,
    FLASH_LORA,
    SCHEDULER_CONFIG_MAP,
    scheduler_names,
    IP_ADAPTER_MODELS,
    IP_ADAPTERS_SD,
    IP_ADAPTERS_SDXL,
    REPO_IMAGE_ENCODER,
    PROMPT_WEIGHT_OPTIONS,
    OLD_PROMPT_WEIGHT_OPTIONS,
    FLASH_AUTO_LOAD_SAMPLER,
)
from .multi_emphasis_prompt import long_prompts_with_weighting
from diffusers.utils import load_image
from .prompt_weights import get_embed_new, add_comma_after_pattern_ti
from .utils import save_pil_image_with_metadata, checkpoint_model_type, get_string_metadata
from .lora_loader import lora_mix_load
from .inpainting_canvas import draw, make_inpaint_condition
from .adetailer import ad_model_process
from ..logging.logging_setup import logger
from .extra_model_loaders import custom_task_model_loader
from .high_resolution import process_images_high_resolution, LATENT_UPSCALERS
from .style_prompt_config import (
    styles_data,
    STYLE_NAMES,
    get_json_content,
    apply_style
)
import os
from compel import Compel, ReturnedEmbeddingsType
import mediapy
from IPython.display import display
from PIL import Image
from typing import Union, Optional, List, Tuple, Dict, Any, Callable # noqa
import logging
import diffusers
import copy
import warnings
import traceback
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
diffusers.utils.logging.set_verbosity(40)
warnings.filterwarnings(action="ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings(action="ignore", category=FutureWarning, module="transformers")

# =====================================
# Utils preprocessor
# =====================================


def resize_image(input_image, resolution, interpolation=None):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / max(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    if interpolation is None:
        interpolation = cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA
    img = cv2.resize(input_image, (W, H), interpolation=interpolation)
    return img


class DepthEstimator:
    def __init__(self):
        self.model = pipeline("depth-estimation")

    def __call__(self, image: np.ndarray, **kwargs) -> PIL.Image.Image:
        detect_resolution = kwargs.pop("detect_resolution", 512)
        image_resolution = kwargs.pop("image_resolution", 512)
        image = np.array(image)
        image = HWC3(image)
        image = resize_image(image, resolution=detect_resolution)
        image = PIL.Image.fromarray(image)
        image = self.model(image)
        image = image["depth"]
        image = np.array(image)
        image = HWC3(image)
        image = resize_image(image, resolution=image_resolution)
        return PIL.Image.fromarray(image)


class ImageSegmentor:
    def __init__(self):
        self.image_processor = AutoImageProcessor.from_pretrained(
            "openmmlab/upernet-convnext-small"
        )
        self.image_segmentor = UperNetForSemanticSegmentation.from_pretrained(
            "openmmlab/upernet-convnext-small"
        )

    @torch.inference_mode()
    def __call__(self, image: np.ndarray, **kwargs) -> PIL.Image.Image:
        detect_resolution = kwargs.pop("detect_resolution", 512)
        image_resolution = kwargs.pop("image_resolution", 512)
        image = HWC3(image)
        image = resize_image(image, resolution=detect_resolution)
        image = PIL.Image.fromarray(image)

        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values
        outputs = self.image_segmentor(pixel_values)
        seg = self.image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(ade_palette()):
            color_seg[seg == label, :] = color
        color_seg = color_seg.astype(np.uint8)

        color_seg = resize_image(
            color_seg, resolution=image_resolution, interpolation=cv2.INTER_NEAREST
        )
        return PIL.Image.fromarray(color_seg)


def apply_gaussian_blur(image_np, ksize=5):
    sigmaX = ksize / 2
    ksize = int(ksize)
    if ksize % 2 == 0:
        ksize += 1
    blurred_image_np = cv2.GaussianBlur(image_np, (ksize, ksize), sigmaX=sigmaX)
    return blurred_image_np


def recolor_luminance(img, thr_a=1.0, **kwargs):
    result = cv2.cvtColor(HWC3(img), cv2.COLOR_BGR2LAB)
    result = result[:, :, 0].astype(np.float32) / 255.0
    result = result ** thr_a
    result = (result * 255.0).clip(0, 255).astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return result


def recolor_intensity(img, thr_a=1.0, **kwargs):
    result = cv2.cvtColor(HWC3(img), cv2.COLOR_BGR2HSV)
    result = result[:, :, 2].astype(np.float32) / 255.0
    result = result ** thr_a
    result = (result * 255.0).clip(0, 255).astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return result


class Preprocessor:
    MODEL_ID = "lllyasviel/Annotators"

    def __init__(self):
        self.model = None
        self.name = ""

    def load(self, name: str) -> None:
        if name == self.name:
            return
        if name == "HED":
            self.model = HEDdetector.from_pretrained(self.MODEL_ID)
        elif name == "Midas":
            self.model = MidasDetector.from_pretrained(self.MODEL_ID)
        elif name == "MLSD":
            self.model = MLSDdetector.from_pretrained(self.MODEL_ID)
        elif name == "Openpose":
            self.model = OpenposeDetector.from_pretrained(self.MODEL_ID)
        elif name == "PidiNet":
            self.model = PidiNetDetector.from_pretrained(self.MODEL_ID)
        elif name == "NormalBae":
            self.model = NormalBaeDetector.from_pretrained(self.MODEL_ID)
        elif name == "Lineart":
            self.model = LineartDetector.from_pretrained(self.MODEL_ID)
        elif name == "LineartAnime":
            self.model = LineartAnimeDetector.from_pretrained(self.MODEL_ID)
        elif name == "Canny":
            self.model = CannyDetector()
        elif name == "ContentShuffle":
            self.model = ContentShuffleDetector()
        elif name == "DPT":
            self.model = DepthEstimator()
        elif name == "UPerNet":
            self.model = ImageSegmentor()
        else:
            raise ValueError
        torch.cuda.empty_cache()
        gc.collect()
        self.name = name

    def __call__(self, image: PIL.Image.Image, **kwargs) -> PIL.Image.Image:
        if self.name == "Canny":
            if "detect_resolution" in kwargs:
                detect_resolution = kwargs.pop("detect_resolution")
                image = np.array(image)
                image = HWC3(image)
                image = resize_image(image, resolution=detect_resolution)
            image = self.model(image, **kwargs)
            return PIL.Image.fromarray(image)
        elif self.name == "Midas":
            detect_resolution = kwargs.pop("detect_resolution", 512)
            image_resolution = kwargs.pop("image_resolution", 512)
            image = np.array(image)
            image = HWC3(image)
            image = resize_image(image, resolution=detect_resolution)
            image = self.model(image, **kwargs)
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            return PIL.Image.fromarray(image)
        else:
            return self.model(image, **kwargs)


# =====================================
# Base Model
# =====================================


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


class PreviewGenerator:
    def __init__(self, *args, **kwargs):
        self.image_step = None
        self.fail_work = None
        self.stream_config(5, 8, False)

    def stream_config(self, concurrency, latent_resize_by, vae_decoding):
        self.concurrency = concurrency
        self.latent_resize_by = latent_resize_by
        self.vae_decoding = vae_decoding

    def decode_tensors(self, pipe, step, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        if step % self.concurrency == 0:  # every how many steps
            logger.debug(step)
            self.image_step = latents_to_rgb(latents, self.latent_resize_by, self.vae_decoding, self.pipe)
            self.new_image_event.set()  # Signal that a new image is available
        return callback_kwargs

    def show_images(self):
        while not self.generation_finished.is_set() or self.new_image_event.is_set():
            self.new_image_event.wait()  # Wait for a new image
            self.new_image_event.clear()  # Clear the event flag

            if self.image_step:
                yield self.image_step  # Yield the new image

            if self.fail_work:
                logger.debug("Stream failed")
                raise Exception(self.fail_work)

    def generate_images(self, pipe_params_config):

        self.final_image = None
        self.image_step = None
        try:
            self.final_image = self.pipe(
                **pipe_params_config,
                callback_on_step_end=self.decode_tensors,
                callback_on_step_end_tensor_inputs=["latents"],
            ).images

            if not isinstance(self.final_image, torch.Tensor):
                self.image_step = self.final_image[0]

            logger.debug("finish")
            self.new_image_event.set()  # Result image
        except Exception as e:
            traceback.print_exc()
            self.fail_work = str(e)
            self.new_image_event.set()

        self.generation_finished.set()  # Signal that generation is finished

    def stream_preview(self, pipe_params_config):

        self.fail_work = None
        self.new_image_event = threading.Event()
        self.generation_finished = threading.Event()

        self.generation_finished.clear()
        threading.Thread(target=self.generate_images, args=(pipe_params_config,)).start()
        return self.show_images()


class Model_Diffusers(PreviewGenerator):
    def __init__(
        self,
        base_model_id: str = "Lykon/dreamshaper-8",
        task_name: str = "txt2img",
        vae_model=None,
        type_model_precision=torch.float16,
        retain_task_model_in_cache=True,
        device=None,
    ):
        super().__init__()
        self.device = (
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )
        self.base_model_id = ""
        self.task_name = ""
        self.vae_model = None
        self.type_model_precision = (
            type_model_precision if torch.cuda.is_available() else torch.float32
        )  # For SD 1.5

        reload = False
        self.load_pipe(
            base_model_id,
            task_name,
            vae_model,
            type_model_precision,
            reload,
            retain_task_model_in_cache,
        )
        self.preprocessor = Preprocessor()

        self.styles_data = styles_data
        self.STYLE_NAMES = STYLE_NAMES
        self.style_json_file = ""

        self.image_encoder_name = None
        self.image_encoder_module = None

    def switch_pipe_class(
        self,
        class_name,
        task_name,
        model_id,
        enable_pag,
    ):

        if hasattr(self.pipe, "set_pag_applied_layers"):
            if not hasattr(self.pipe, "text_encoder_2"):
                self.pipe = StableDiffusionPipeline(
                    vae=self.pipe.vae,
                    text_encoder=self.pipe.text_encoder,
                    tokenizer=self.pipe.tokenizer,
                    unet=self.pipe.unet,
                    scheduler=self.pipe.scheduler,
                    safety_checker=self.pipe.safety_checker,
                    feature_extractor=self.pipe.feature_extractor,
                    image_encoder=self.pipe.image_encoder,
                    requires_safety_checker=self.pipe.config.requires_safety_checker,
                )
            else:
                self.pipe = StableDiffusionXLPipeline(
                    vae=self.pipe.vae,
                    text_encoder=self.pipe.text_encoder,
                    text_encoder_2=self.pipe.text_encoder_2,
                    tokenizer=self.pipe.tokenizer,
                    tokenizer_2=self.pipe.tokenizer_2,
                    unet=self.pipe.unet,
                    scheduler=self.pipe.scheduler,
                    feature_extractor=self.pipe.feature_extractor,
                    image_encoder=self.pipe.image_encoder,
                )

        tk = "base"
        model_components = dict(
            vae=self.pipe.vae,
            text_encoder=self.pipe.text_encoder,
            tokenizer=self.pipe.tokenizer,
            unet=self.pipe.unet,
            scheduler=self.pipe.scheduler,
            feature_extractor=self.pipe.feature_extractor,
            image_encoder=self.pipe.image_encoder,
        )
        if class_name == "StableDiffusionPipeline":
            model_components["safety_checker"] = self.pipe.safety_checker
            model_components["requires_safety_checker"] = self.pipe.config.requires_safety_checker

            if task_name not in ["txt2img", "img2img"]:
                model_components["controlnet"] = ControlNetModel.from_pretrained(
                    model_id, torch_dtype=self.type_model_precision
                )
                tk = "controlnet"

        elif class_name == "StableDiffusionXLPipeline":
            model_components["text_encoder_2"] = self.pipe.text_encoder_2
            model_components["tokenizer_2"] = self.pipe.tokenizer_2

            if task_name not in ["txt2img", "inpaint", "img2img"]:
                if "t2i" not in task_name:
                    model_components["controlnet"] = ControlNetModel.from_pretrained(
                        model_id, torch_dtype=torch.float16, variant="fp16"
                    ).to(self.device)
                    tk = "controlnet"
                else:
                    model_components["adapter"] = T2IAdapter.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16,
                        varient="fp16",
                    ).to(self.device)
                    tk = "adapter"

        if task_name == "inpaint":
            tk = "inpaint"

        if enable_pag:
            if (
                tk == "adapter" or
                (task_name in ["inpaint", "img2img"] and "XL" not in class_name)
            ):
                logger.warning(
                    f"PAG is not enabled for {class_name} with {task_name}."
                )
                enable_pag = False

        # Load Pipeline
        if enable_pag:
            model_components["pag_applied_layers"] = "mid"
            self.pipe = CLASS_PAG_DIFFUSERS_TASK[class_name][tk](**model_components).to(self.device)
        else:
            self.pipe = CLASS_DIFFUSERS_TASK[class_name][tk](**model_components).to(self.device)

        if task_name == "img2img":
            self.pipe = AutoPipelineForImage2Image.from_pipe(self.pipe, enable_pag=enable_pag)

        # Create new base values
        self.pipe.to(self.device)
        torch.cuda.empty_cache()
        gc.collect()

    def load_pipe(
        self,
        base_model_id: str,
        task_name="txt2img",
        vae_model=None,
        type_model_precision=torch.float16,
        reload=False,
        retain_task_model_in_cache=True,
    ) -> DiffusionPipeline:
        if (
            base_model_id == self.base_model_id
            and task_name == self.task_name
            and hasattr(self, "pipe")
            and self.vae_model == vae_model
            and self.pipe is not None
            and reload is False
        ):
            if self.type_model_precision == type_model_precision or self.device.type == "cpu":
                return

        if hasattr(self, "pipe") and os.path.isfile(base_model_id):
            unload_model = False
            if self.pipe is None:
                unload_model = True
            elif type_model_precision != self.type_model_precision and self.device.type != "cpu":
                unload_model = True
        else:
            if hasattr(self, "pipe"):
                unload_model = False
                if self.pipe is None:
                    unload_model = True
            else:
                unload_model = True
        self.type_model_precision = (
            type_model_precision if torch.cuda.is_available() else torch.float32
        )

        if self.type_model_precision == torch.float32 and os.path.isfile(base_model_id):
            logger.info(f"Working with full precision {str(self.type_model_precision)}")

        # Load model
        if self.base_model_id == base_model_id and self.pipe is not None and reload is False and self.vae_model == vae_model and unload_model is False:
            # logger.info("Previous loaded base model") # not return
            class_name = self.class_name
        else:
            # Unload previous model and stuffs
            self.pipe = None
            self.task_name = ""
            self.model_memory = {}
            self.lora_memory = [None, None, None, None, None]
            self.lora_scale_memory = [1.0, 1.0, 1.0, 1.0, 1.0]
            self.lora_status = [None] * 5
            self.flash_config = None
            self.ip_adapter_config = None
            self.embed_loaded = []
            self.FreeU = False
            torch.cuda.empty_cache()
            gc.collect()

            # Load new model
            if os.path.isfile(base_model_id):  # exists or not same # if os.path.exists(base_model_id):

                if base_model_id.endswith(".safetensors"):
                    model_type = checkpoint_model_type(base_model_id)
                    logger.debug(f"Infered model type is {model_type}")
                else:
                    model_type = "sd1.5"

                if model_type == "sdxl":
                    logger.info("Default VAE: madebyollin/sdxl-vae-fp16-fix")
                    self.pipe = StableDiffusionXLPipeline.from_single_file(
                        base_model_id,
                        vae=AutoencoderKL.from_pretrained(
                            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
                        ),
                        torch_dtype=self.type_model_precision,
                    )
                    class_name = "StableDiffusionXLPipeline"
                elif model_type == "sd1.5":
                    self.pipe = StableDiffusionPipeline.from_single_file(
                        base_model_id,
                        # vae=None
                        # if vae_model == None
                        # else AutoencoderKL.from_single_file(
                        #     vae_model
                        # ),
                        torch_dtype=self.type_model_precision,
                    )
                    class_name = "StableDiffusionPipeline"
                else:
                    raise ValueError(f"Model type {model_type} not supported.")
            else:
                file_config = hf_hub_download(repo_id=base_model_id, filename="model_index.json")

                # Reading data from the JSON file
                with open(file_config, 'r') as json_config:
                    data_config = json.load(json_config)

                # Searching for the value of the "_class_name" key
                if '_class_name' in data_config:
                    class_name = data_config['_class_name']

                match class_name:
                    case "StableDiffusionPipeline":
                        self.pipe = StableDiffusionPipeline.from_pretrained(
                            base_model_id,
                            torch_dtype=self.type_model_precision,
                        )

                    case "StableDiffusionXLPipeline":
                        logger.info("Default VAE: madebyollin/sdxl-vae-fp16-fix")
                        try:
                            self.pipe = DiffusionPipeline.from_pretrained(
                                base_model_id,
                                vae=AutoencoderKL.from_pretrained(
                                    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
                                ),
                                torch_dtype=torch.float16,
                                use_safetensors=True,
                                variant="fp16",
                                add_watermarker=False,
                            )
                        except Exception as e:
                            logger.debug(e)
                            logger.debug("Loading model without parameter variant=fp16")
                            self.pipe = DiffusionPipeline.from_pretrained(
                                base_model_id,
                                vae=AutoencoderKL.from_pretrained(
                                    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
                                ),
                                torch_dtype=torch.float16,
                                use_safetensors=True,
                                add_watermarker=False,
                            )
            self.base_model_id = base_model_id
            self.class_name = class_name

            # Load VAE after loaded model
            if vae_model is None:
                logger.debug("Default VAE")
                pass
            else:
                if os.path.isfile(vae_model):
                    self.pipe.vae = AutoencoderKL.from_single_file(
                        vae_model
                    )
                else:
                    self.pipe.vae = AutoencoderKL.from_pretrained(
                        vae_model,
                        subfolder="vae",
                    )
                try:
                    self.pipe.vae.to(self.type_model_precision)
                except Exception as e:
                    logger.debug(str(e))
                    logger.warning(f"VAE: not in {self.type_model_precision}")
            self.vae_model = vae_model

            # Define base scheduler
            self.default_scheduler = copy.deepcopy(self.pipe.scheduler)
            logger.debug(f"Base sampler: {self.default_scheduler}")

        if task_name in self.model_memory:
            self.pipe = self.model_memory[task_name]
            # Create new base values
            # self.pipe.to(self.device)
            # torch.cuda.empty_cache()
            # gc.collect()
            self.base_model_id = base_model_id
            self.task_name = task_name
            self.vae_model = vae_model
            self.class_name = class_name
            self.pipe.watermark = None
            return

        # if class_name == "StableDiffusionPipeline" and task_name not in SD15_TASKS:
        #     logger.error(f"The selected task: {task_name} is not implemented for SD 1.5")
        # elif class_name == "StableDiffusionXLPipeline" and task_name not in SDXL_TASKS:
        #     logger.error(f"The selected task: {task_name} is not implemented for SDXL")

        # Load task
        model_id = CONTROLNET_MODEL_IDS[task_name]
        if isinstance(model_id, list):
            if "XL" in class_name:
                model_id = model_id[1]
            else:
                model_id = model_id[0]

        if (
            (self.task_name != task_name)
            or (self.class_name != class_name)
        ):
            self.switch_pipe_class(
                class_name,
                task_name,
                model_id,
                enable_pag=False,
            )

        self.model_id_task = model_id

        self.base_model_id = base_model_id
        self.task_name = task_name
        self.vae_model = vae_model
        self.class_name = class_name

        if self.class_name == "StableDiffusionXLPipeline":
            self.pipe.enable_vae_slicing()
            self.pipe.enable_vae_tiling()
            self.pipe.watermark = None

        if retain_task_model_in_cache is True and task_name not in self.model_memory:
            self.model_memory[task_name] = self.pipe

        return

    def load_controlnet_weight(self, task_name: str) -> None:
        torch.cuda.empty_cache()
        gc.collect()
        model_id = CONTROLNET_MODEL_IDS[task_name]
        if isinstance(model_id, list):
            # SD1.5 model
            model_id = model_id[0]
        controlnet = ControlNetModel.from_pretrained(
            model_id, torch_dtype=self.type_model_precision
        )
        controlnet.to(self.device)
        torch.cuda.empty_cache()
        gc.collect()
        self.pipe.controlnet = controlnet
        # self.task_name = task_name

    @torch.inference_mode()
    def get_image_preprocess(
        self,
        image: np.ndarray,
        image_resolution: int,
        preprocess_resolution: int,
        low_threshold: int,
        high_threshold: int,
        preprocessor_name: str,
        value_threshold: float,
        distance_threshold: float,
        t2i_adapter_preprocessor: bool,
        recolor_gamma_correction: float,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError("No reference image found.")

        # if self.class_name == "StableDiffusionPipeline" and self.task_name in ["lineart", "lineart_anime"]:
        #     if "anime" in preprocessor_name:
        #         self.load_controlnet_weight("lineart_anime")
        #         logger.info("Linear anime")
        #     else:
        #         self.load_controlnet_weight("lineart")

        if "t2i" in self.task_name:
            preprocessor_name = T2I_PREPROCESSOR_NAME[self.task_name] if t2i_adapter_preprocessor else "None"

        params_preprocessor = {
            "image": image,
            "image_resolution": image_resolution,
            "detect_resolution": preprocess_resolution,
        }

        if preprocessor_name in ["None", "None (anime)"] or self.task_name in ["ip2p", "img2img", "pattern", "sdxl_tile_realistic"]:
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        elif self.task_name in ["canny", "sdxl_canny_t2i"]:
            self.preprocessor.load("Canny")
            control_image = self.preprocessor(
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                **params_preprocessor
            )
        elif self.task_name in ["openpose", "sdxl_openpose_t2i"]:
            self.preprocessor.load("Openpose")
            control_image = self.preprocessor(
                hand_and_face=True,
                **params_preprocessor
            )
        elif self.task_name in ["depth", "sdxl_depth-midas_t2i"]:
            self.preprocessor.load(preprocessor_name)
            control_image = self.preprocessor(
                **params_preprocessor
            )
        elif self.task_name == "mlsd":
            self.preprocessor.load("MLSD")
            control_image = self.preprocessor(
                thr_v=value_threshold,
                thr_d=distance_threshold,
                **params_preprocessor
            )
        elif self.task_name in ["scribble", "sdxl_sketch_t2i"]:
            if preprocessor_name == "HED":
                self.preprocessor.load(preprocessor_name)
                control_image = self.preprocessor(
                    scribble=False,
                    **params_preprocessor
                )
            elif preprocessor_name == "PidiNet":
                self.preprocessor.load(preprocessor_name)
                control_image = self.preprocessor(
                    safe=False,
                    **params_preprocessor
                )
        elif self.task_name == "softedge":
            if preprocessor_name in ["HED", "HED safe"]:
                safe = "safe" in preprocessor_name
                self.preprocessor.load("HED")
                control_image = self.preprocessor(
                    scribble=safe,
                    **params_preprocessor
                )
            elif preprocessor_name in ["PidiNet", "PidiNet safe"]:
                safe = "safe" in preprocessor_name
                self.preprocessor.load("PidiNet")
                control_image = self.preprocessor(
                    safe=safe,
                    **params_preprocessor
                )
        elif self.task_name == "segmentation":
            self.preprocessor.load(preprocessor_name)
            control_image = self.preprocessor(
                **params_preprocessor
            )
        elif self.task_name == "normalbae":
            self.preprocessor.load("NormalBae")
            control_image = self.preprocessor(
                **params_preprocessor
            )
        elif self.task_name in ["lineart", "lineart_anime", "sdxl_lineart_t2i"]:
            if preprocessor_name in ["Lineart", "Lineart coarse"]:
                coarse = "coarse" in preprocessor_name
                self.preprocessor.load("Lineart")
                control_image = self.preprocessor(
                    coarse=coarse,
                    **params_preprocessor
                )
            elif preprocessor_name == "Lineart (anime)":
                self.preprocessor.load("LineartAnime")
                control_image = self.preprocessor(
                    **params_preprocessor
                )
        elif self.task_name == "shuffle":
            self.preprocessor.load(preprocessor_name)
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
            )
        elif self.task_name == "tile":
            image_np = resize_image(image, resolution=image_resolution)
            blur_names = {
                "Mild Blur": 5,
                "Moderate Blur": 15,
                "Heavy Blur": 27,
            }
            image_np = apply_gaussian_blur(
                image_np, ksize=blur_names[preprocessor_name]
            )
            control_image = PIL.Image.fromarray(image_np)
        elif self.task_name == "recolor":
            image_np = resize_image(image, resolution=image_resolution)

            if preprocessor_name == "Recolor luminance":
                image_np = recolor_luminance(image_np, thr_a=recolor_gamma_correction)
            elif preprocessor_name == "Recolor intensity":
                image_np = recolor_intensity(image_np, thr_a=recolor_gamma_correction)

            control_image = PIL.Image.fromarray(image_np)
        else:
            raise ValueError("No valid preprocessor name")

        return control_image

    @torch.inference_mode()
    def process_inpaint(
        self,
        image: np.ndarray,
        image_resolution: int,
        image_mask: str,
        gui_active: bool,
    ) -> list[PIL.Image.Image]:

        # Get mask for Inpaint
        if gui_active or image_mask:
            array_rgb_mask = convert_image_to_numpy_array(image_mask, gui_active)
        elif not gui_active:
            # Convert control image to draw
            import base64
            import matplotlib.pyplot as plt

            Image.fromarray(image).save("inpaint_image.png")
            image_aux = "./inpaint_image.png"

            name_without_extension = os.path.splitext(image_aux.split("/")[-1])[0]
            image64 = base64.b64encode(open(image_aux, "rb").read())
            image64 = image64.decode("utf-8")
            img = np.array(plt.imread(f"{image_aux}")[:, :, :3])

            # Create mask interactive
            logger.info("Draw the mask on this canvas using the mouse. When you finish, press 'Finish' in the bottom side of the canvas.")
            draw(
                image64,
                filename=f"./{name_without_extension}_draw.png",
                w=img.shape[1],
                h=img.shape[0],
                line_width=0.04 * img.shape[1],
            )

            # Create mask and save
            with_mask = np.array(
                plt.imread(f"./{name_without_extension}_draw.png")[:, :, :3]
            )
            mask = (
                (with_mask[:, :, 0] == 1)
                * (with_mask[:, :, 1] == 0)
                * (with_mask[:, :, 2] == 0)
            )
            plt.imsave(f"./{name_without_extension}_mask.png", mask, cmap="gray")
            mask_control = f"./{name_without_extension}_mask.png"
            logger.info(f"Mask saved: {mask_control}")

            # Read image mask
            array_rgb_mask = convert_image_to_numpy_array(mask_control, gui_active)
        else:
            raise ValueError("No image mask was found.")

        image = HWC3(image)
        image = resize_image(image, resolution=image_resolution)
        init_image = PIL.Image.fromarray(image)

        image_mask_hwc = HWC3(array_rgb_mask)
        image_mask = resize_image(image_mask_hwc, resolution=image_resolution)
        control_mask = PIL.Image.fromarray(image_mask)

        control_image = make_inpaint_condition(init_image, control_mask)

        return init_image, control_mask, control_image

    def get_scheduler(self, name):
        if name in SCHEDULER_CONFIG_MAP:
            scheduler_class, config = SCHEDULER_CONFIG_MAP[name]
            # return scheduler_class.from_config(self.pipe.scheduler.config, **config)
            # beta self.default_scheduler
            return scheduler_class.from_config(self.default_scheduler.config, **config)
        else:
            raise ValueError(f"Scheduler with name {name} not found. Valid schedulers: {', '.join(scheduler_names)}")

    def emphasis_prompt(
        self,
        pipe,
        prompt,
        negative_prompt,
        clip_skip=2,  # disabled with sdxl
        emphasis="Original",
        comma_padding_backtrack=20,
    ):

        if hasattr(pipe, "text_encoder_2"):
            # Prompt weights for textual inversion
            try:
                prompt_ti = pipe.maybe_convert_prompt(prompt, pipe.tokenizer)
                negative_prompt_ti = pipe.maybe_convert_prompt(negative_prompt, pipe.tokenizer)
            except Exception as e:
                logger.debug(str(e))
                prompt_ti = prompt
                negative_prompt_ti = negative_prompt
                logger.error("FAILED: Convert prompt for textual inversion")

            cond, uncond = long_prompts_with_weighting(
                pipe,
                prompt_ti,
                negative_prompt_ti,
                clip_skip=clip_skip,  # disabled with sdxl
                emphasis=emphasis,
                comma_padding_backtrack=comma_padding_backtrack
            )

            cond_tensor = [cond[0], uncond[0]]
            all_cond = torch.cat(cond_tensor)

            pooled_tensor = [cond[1], uncond[1]]
            all_pooled = torch.cat(pooled_tensor)

            assert torch.equal(all_cond[0:1], cond[0]), "Tensors are not equal"

            return all_cond, all_pooled
        else:
            # Prompt weights for textual inversion
            prompt_ti = self.pipe.maybe_convert_prompt(prompt, self.pipe.tokenizer)
            negative_prompt_ti = self.pipe.maybe_convert_prompt(
                negative_prompt, self.pipe.tokenizer
            )

            # separate the multi-vector textual inversion by comma
            if self.embed_loaded != []:
                prompt_ti = add_comma_after_pattern_ti(prompt_ti)
                negative_prompt_ti = add_comma_after_pattern_ti(negative_prompt_ti)

            return long_prompts_with_weighting(
                pipe,
                prompt_ti,
                negative_prompt_ti,
                clip_skip=clip_skip,
                emphasis=emphasis,
                comma_padding_backtrack=comma_padding_backtrack
            )

    def create_prompt_embeds(
        self,
        prompt,
        negative_prompt,
        textual_inversion,
        clip_skip,
        syntax_weights,
    ):
        if self.class_name == "StableDiffusionPipeline":
            if self.embed_loaded != textual_inversion and textual_inversion != []:
                # Textual Inversion
                for name, directory_name in textual_inversion:
                    try:
                        if directory_name.endswith(".pt"):
                            model = torch.load(directory_name, map_location=self.device)
                            model_tensors = model.get("string_to_param").get("*")
                            s_model = {"emb_params": model_tensors}
                            # save_file(s_model, directory_name[:-3] + '.safetensors')
                            self.pipe.load_textual_inversion(s_model, token=name)

                        else:
                            # self.pipe.text_encoder.resize_token_embeddings(len(self.pipe.tokenizer),pad_to_multiple_of=128)
                            # self.pipe.load_textual_inversion("./bad_prompt.pt", token="baddd")
                            self.pipe.load_textual_inversion(directory_name, token=name)
                        if not self.gui_active:
                            logger.info(f"Applied : {name}")

                    except Exception as e:
                        exception = str(e)
                        if name in exception:
                            logger.debug(f"Previous loaded embed {name}")
                        else:
                            logger.error(exception)
                            logger.error(f"Can't apply embed {name}")
                self.embed_loaded = textual_inversion

            if syntax_weights not in OLD_PROMPT_WEIGHT_OPTIONS:
                emphasis = PROMPT_WEIGHT_OPTIONS[syntax_weights]
                return self.emphasis_prompt(
                    self.pipe,
                    prompt,
                    negative_prompt,
                    clip_skip=2 if clip_skip else 1,  # disabled with sdxl
                    emphasis=emphasis,
                    comma_padding_backtrack=20,
                )

            # Clip skip
            # clip_skip_diffusers = None #clip_skip - 1 # future update
            if not hasattr(self, "compel"):
                self.compel = Compel(
                    tokenizer=self.pipe.tokenizer,
                    text_encoder=self.pipe.text_encoder,
                    truncate_long_prompts=False,
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED if clip_skip else ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
                )

            # Prompt weights for textual inversion
            prompt_ti = self.pipe.maybe_convert_prompt(prompt, self.pipe.tokenizer)
            negative_prompt_ti = self.pipe.maybe_convert_prompt(
                negative_prompt, self.pipe.tokenizer
            )

            # separate the multi-vector textual inversion by comma
            if self.embed_loaded != []:
                prompt_ti = add_comma_after_pattern_ti(prompt_ti)
                negative_prompt_ti = add_comma_after_pattern_ti(negative_prompt_ti)

            # Syntax weights
            self.pipe.to(self.device)
            if syntax_weights == "Classic":
                prompt_emb = get_embed_new(prompt_ti, self.pipe, self.compel)
                negative_prompt_emb = get_embed_new(negative_prompt_ti, self.pipe, self.compel)
            else:
                prompt_emb = get_embed_new(prompt_ti, self.pipe, self.compel, compel_process_sd=True)
                negative_prompt_emb = get_embed_new(negative_prompt_ti, self.pipe, self.compel, compel_process_sd=True)

            # Fix error shape
            if prompt_emb.shape != negative_prompt_emb.shape:
                (
                    prompt_emb,
                    negative_prompt_emb,
                ) = self.compel.pad_conditioning_tensors_to_same_length(
                    [prompt_emb, negative_prompt_emb]
                )

            return prompt_emb, negative_prompt_emb

        else:
            # SDXL embed
            if self.embed_loaded != textual_inversion and textual_inversion != []:
                # Textual Inversion
                for name, directory_name in textual_inversion:
                    try:
                        from safetensors.torch import load_file
                        state_dict = load_file(directory_name)
                        self.pipe.load_textual_inversion(state_dict["clip_g"], token=name, text_encoder=self.pipe.text_encoder_2, tokenizer=self.pipe.tokenizer_2)
                        self.pipe.load_textual_inversion(state_dict["clip_l"], token=name, text_encoder=self.pipe.text_encoder, tokenizer=self.pipe.tokenizer)
                        if not self.gui_active:
                            logger.info(f"Applied : {name}")
                    except Exception as e:
                        exception = str(e)
                        if name in exception:
                            logger.debug(f"Previous loaded embed {name}")
                        else:
                            logger.error(exception)
                            logger.error(f"Can't apply embed {name}")
                self.embed_loaded = textual_inversion

            if syntax_weights not in OLD_PROMPT_WEIGHT_OPTIONS:
                emphasis = PROMPT_WEIGHT_OPTIONS[syntax_weights]
                return self.emphasis_prompt(
                    self.pipe,
                    prompt,
                    negative_prompt,
                    clip_skip=2 if clip_skip else 1,  # disabled with sdxl
                    emphasis=emphasis,
                    comma_padding_backtrack=20,
                )

            if not hasattr(self, "compel"):
                # Clip skip
                if clip_skip:
                    # clip_skip_diffusers = None #clip_skip - 1 # future update
                    self.compel = Compel(
                        tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
                        text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
                        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                        requires_pooled=[False, True],
                        truncate_long_prompts=False,
                    )
                else:
                    # clip_skip_diffusers = None # clip_skip = None # future update
                    self.compel = Compel(
                        tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
                        text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
                        requires_pooled=[False, True],
                        truncate_long_prompts=False,
                    )

            # Prompt weights for textual inversion
            try:
                prompt_ti = self.pipe.maybe_convert_prompt(prompt, self.pipe.tokenizer)
                negative_prompt_ti = self.pipe.maybe_convert_prompt(negative_prompt, self.pipe.tokenizer)
            except Exception as e:
                logger.debug(str(e))
                prompt_ti = prompt
                negative_prompt_ti = negative_prompt
                logger.error("FAILED: Convert prompt for textual inversion")

            # prompt syntax style a1...
            if syntax_weights == "Classic":
                # self.pipe.to("cuda")
                prompt_ti = get_embed_new(prompt_ti, self.pipe, self.compel, only_convert_string=True)
                negative_prompt_ti = get_embed_new(negative_prompt_ti, self.pipe, self.compel, only_convert_string=True)
            else:
                prompt_ti = prompt
                negative_prompt_ti = negative_prompt

            conditioning, pooled = self.compel([prompt_ti, negative_prompt_ti])

            return conditioning, pooled

    def process_lora(self, select_lora, lora_weights_scale, unload=False):
        device = "cuda" if self.device.type != "cpu" else "cpu"
        status_lora = None
        if not unload:
            if select_lora is not None:
                status_lora = True
                try:
                    self.pipe = lora_mix_load(
                        self.pipe,
                        select_lora,
                        lora_weights_scale,
                        device=device,
                        dtype=self.type_model_precision,
                    )
                    logger.info(select_lora)
                except Exception as e:
                    if logger.isEnabledFor(logging.DEBUG):
                        traceback.print_exc()
                    logger.error(f"ERROR: LoRA not compatible: {select_lora}")
                    logger.debug(f"{str(e)}")
                    status_lora = False
        else:
            # Unload numerically unstable but fast and need less memory
            if select_lora is not None:
                try:
                    self.pipe = lora_mix_load(
                        self.pipe,
                        select_lora,
                        -lora_weights_scale,
                        device=device,
                        dtype=self.type_model_precision,
                    )
                    logger.debug(f"Unload LoRA: {select_lora}")
                except Exception as e:
                    logger.debug(str(e))
                    pass

        return status_lora

    def load_style_file(self, style_json_file):
        if os.path.exists(style_json_file):
            try:
                file_json_read = get_json_content(style_json_file)
                self.styles_data = {
                    k["name"]: (k["prompt"], k["negative_prompt"] if "negative_prompt" in k else "") for k in file_json_read
                }
                self.STYLE_NAMES = list(self.styles_data.keys())
                self.style_json_file = style_json_file
                logger.info(
                    f"Styles json file loaded with {len(self.STYLE_NAMES)} styles"
                )
                logger.debug(str(self.STYLE_NAMES))
            except Exception as e:
                logger.error(str(e))
        else:
            logger.error("Not found styles json file in directory")

    def load_beta_styles(self):
        from .constants import BETA_STYLE_LIST

        styles_data = {
            k["name"]: (k["prompt"], k["negative_prompt"] if "negative_prompt" in k else "") for k in BETA_STYLE_LIST
        }
        STYLE_NAMES = list(styles_data.keys())
        self.styles_data = styles_data
        self.STYLE_NAMES = STYLE_NAMES
        self.style_json_file = ""

        logger.info(
                    f"Beta styles loaded with {len(self.STYLE_NAMES)} styles"
                )

    def set_ip_adapter_multimode_scale(self, ip_scales, ip_adapter_mode):
        mode_scales = []
        for scale, mode in zip(ip_scales, ip_adapter_mode):
            if mode == "style":
                map_scale = {
                    "up": {"block_0": [0.0, scale, 0.0]},
                }
            elif mode == "layout":
                map_scale = {
                    "down": {"block_2": [0.0, scale]},
                }
            elif mode == "style+layout":
                map_scale = {
                    "down": {"block_2": [0.0, scale]},
                    "up": {"block_0": [0.0, scale, 0.0]},
                }
            else:
                map_scale = scale

            mode_scales.append(map_scale)

        self.pipe.set_ip_adapter_scale(mode_scales)

    def set_ip_adapter_model(self, ip_weights):

        repo_name = [data[0] for data in ip_weights]
        sub_folder = [data[1] for data in ip_weights]
        weight_name = [data[2] for data in ip_weights]
        vit_model = [data[3] for data in ip_weights]
        vit_model = list(set(x for x in vit_model if x is not None))

        if len(vit_model) > 1:
            raise ValueError("Can't combine vit-G with vit-H models")

        all_contain_faceid = all("faceid" in m for m in weight_name)
        all_not_contain_faceid = all("faceid" not in m for m in weight_name)

        if not (all_contain_faceid or all_not_contain_faceid):
            raise ValueError("Can't combine ip adapters with faceid adapters")

        if (
            (vit_model and self.image_encoder_name != vit_model[0])
            or (
                vit_model
                and self.image_encoder_name == vit_model[0]
                and (getattr(self.pipe, "image_encoder", None) is None)
            )
        ):
            from transformers import CLIPVisionModelWithProjection

            vit_repo = REPO_IMAGE_ENCODER[vit_model[0]]

            self.image_encoder_module = CLIPVisionModelWithProjection.from_pretrained(
                vit_repo,
                torch_dtype=self.type_model_precision,
            ).to(self.device)
            self.image_encoder_name = vit_model[0]
            self.pipe.register_modules(image_encoder=self.image_encoder_module)  # automatic change

        # Load ip_adapter
        self.pipe.load_ip_adapter(
            repo_name,
            subfolder=sub_folder,
            weight_name=weight_name,
            image_encoder_folder=None
        )

        self.ip_adapter_config = weight_name  # change to key dict ipmodels

    def get_ip_embeds(
            self, guidance_scale, ip_images, num_images, ip_masks=None
    ):
        do_classifier_free_guidance = guidance_scale > 1

        if "faceid" not in self.ip_adapter_config[0]:
            with torch.no_grad():
                image_embeds = self.pipe.prepare_ip_adapter_image_embeds(
                    ip_images,
                    None,
                    self.device,
                    num_images,
                    do_classifier_free_guidance,
                )  # is a list
        else:
            from insightface.app import FaceAnalysis
            from insightface.utils import face_align
            from IPython.utils.capture import capture_output

            with capture_output() as captured:  # noqa
                app = FaceAnalysis(  # cache this
                    name="buffalo_l",
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                app.prepare(ctx_id=0, det_size=(640, 640))

            image_embeds = []
            for i, (image, ip_weight) in enumerate(zip(ip_images, self.ip_adapter_config)):

                if not isinstance(image, list):
                    image = [image]

                image_embeds_single = []
                image_projection = []
                for j, single_image in enumerate(image):

                    single_image = cv2.cvtColor(np.asarray(single_image), cv2.COLOR_BGR2RGB)
                    faces = app.get(single_image)

                    if len(faces) == 0:
                        num_batch_image = "" if len(image) == 1 else f", subimage {j+1}"
                        raise ValueError(f"No face detected in image number {i+1}{num_batch_image}")

                    if "plus" in ip_weight:
                        face_crop_align = face_align.norm_crop(single_image, landmark=faces[0].kps, image_size=224)
                        image_projection.append(face_crop_align)

                    single_image = torch.from_numpy(faces[0].normed_embedding)
                    ref_images_embeds = []
                    ref_images_embeds.append(single_image.unsqueeze(0))
                    ref_images_embeds = torch.stack(ref_images_embeds, dim=0).unsqueeze(0)

                    neg_ref_images_embeds = torch.zeros_like(ref_images_embeds)

                    id_embed = torch.cat([neg_ref_images_embeds, ref_images_embeds]).to(dtype=self.type_model_precision, device=self.device)
                    image_embeds_single.append(id_embed)

                image_embeds.append(torch.cat(image_embeds_single, dim=1))

                if image_projection:
                    clip_embeds = self.pipe.prepare_ip_adapter_image_embeds(
                        [image_projection] * len(ip_images),
                        None,
                        torch.device(self.device),
                        num_images,
                        do_classifier_free_guidance
                    )[0]

                    gc.collect()
                    torch.cuda.empty_cache()

                    self.pipe.unet.encoder_hid_proj.image_projection_layers[i].clip_embeds = clip_embeds.to(dtype=self.type_model_precision)
                    if "plusv2" in ip_weight:
                        self.pipe.unet.encoder_hid_proj.image_projection_layers[i].shortcut = True
                    else:
                        self.pipe.unet.encoder_hid_proj.image_projection_layers[i].shortcut = False

                gc.collect()
                torch.cuda.empty_cache()

            # average_embedding = torch.mean(torch.stack(faceid_all_embeds, dim=0), dim=0)

        processed_masks = []
        if ip_masks and ip_masks[0] is not None:  # fix this auto generate mask if any have it...
            from diffusers.image_processor import IPAdapterMaskProcessor

            processor = IPAdapterMaskProcessor()
            first_mask = ip_masks[0]
            if isinstance(first_mask, list):
                first_mask = first_mask[0]
            width, height = first_mask.size  # aspect ratio based on first mask

            for mask in ip_masks:
                if not isinstance(mask, list):
                    mask = [mask]
                masks_ = processor.preprocess(mask, height=height, width=width)

                if len(mask) > 1:
                    masks_ = [masks_.reshape(1, masks_.shape[0], masks_.shape[2], masks_.shape[3])]

                processed_masks.append(masks_)

        return image_embeds, processed_masks

    def lora_merge(
        self,
        lora_A=None, lora_scale_A=1.0,
        lora_B=None, lora_scale_B=1.0,
        lora_C=None, lora_scale_C=1.0,
        lora_D=None, lora_scale_D=1.0,
        lora_E=None, lora_scale_E=1.0,
    ):

        lora_status = [None] * 5

        if self.lora_memory == [
            lora_A,
            lora_B,
            lora_C,
            lora_D,
            lora_E,
        ] and self.lora_scale_memory == [
            lora_scale_A,
            lora_scale_B,
            lora_scale_C,
            lora_scale_D,
            lora_scale_E,
        ]:
            for single_lora in self.lora_memory:
                if single_lora is not None:
                    logger.info(f"LoRA in memory: {single_lora}")
            pass

        else:
            logger.debug("_un, re and load_ lora")

            self.process_lora(
                self.lora_memory[0], self.lora_scale_memory[0], unload=True
            )
            self.process_lora(
                self.lora_memory[1], self.lora_scale_memory[1], unload=True
            )
            self.process_lora(
                self.lora_memory[2], self.lora_scale_memory[2], unload=True
            )
            self.process_lora(
                self.lora_memory[3], self.lora_scale_memory[3], unload=True
            )
            self.process_lora(
                self.lora_memory[4], self.lora_scale_memory[4], unload=True
            )

            lora_status[0] = self.process_lora(lora_A, lora_scale_A)
            lora_status[1] = self.process_lora(lora_B, lora_scale_B)
            lora_status[2] = self.process_lora(lora_C, lora_scale_C)
            lora_status[3] = self.process_lora(lora_D, lora_scale_D)
            lora_status[4] = self.process_lora(lora_E, lora_scale_E)

        self.lora_memory = [lora_A, lora_B, lora_C, lora_D, lora_E]
        self.lora_scale_memory = [
            lora_scale_A,
            lora_scale_B,
            lora_scale_C,
            lora_scale_D,
            lora_scale_E,
        ]

        return lora_status

    def __call__(
        self,
        prompt: str = "",
        negative_prompt: str = "",
        img_height: int = 512,
        img_width: int = 512,
        num_images: int = 1,
        num_steps: int = 30,
        guidance_scale: float = 7.5,
        clip_skip: Optional[bool] = True,
        seed: int = -1,
        sampler: str = "DPM++ 2M",
        syntax_weights: str = "Classic",

        lora_A: Optional[str] = None,
        lora_scale_A: float = 1.0,
        lora_B: Optional[str] = None,
        lora_scale_B: float = 1.0,
        lora_C: Optional[str] = None,
        lora_scale_C: float = 1.0,
        lora_D: Optional[str] = None,
        lora_scale_D: float = 1.0,
        lora_E: Optional[str] = None,
        lora_scale_E: float = 1.0,
        textual_inversion: List[Tuple[str, str]] = [],
        FreeU: bool = False,
        adetailer_A: bool = False,
        adetailer_A_params: Dict[str, Any] = {},
        adetailer_B: bool = False,
        adetailer_B_params: Dict[str, Any] = {},
        style_prompt: Optional[Any] = [""],
        style_json_file: Optional[Any] = "",
        pag_scale: float = 0.,

        image: Optional[Any] = None,
        preprocessor_name: Optional[str] = "None",
        preprocess_resolution: int = 512,
        image_resolution: int = 512,
        image_mask: Optional[Any] = None,
        strength: float = 0.35,
        low_threshold: int = 100,
        high_threshold: int = 200,
        value_threshold: float = 0.1,
        distance_threshold: float = 0.1,
        recolor_gamma_correction: float = 1.0,
        controlnet_conditioning_scale: float = 1.0,
        control_guidance_start: float = 0.0,
        control_guidance_end: float = 1.0,
        t2i_adapter_preprocessor: bool = True,
        t2i_adapter_conditioning_scale: float = 1.0,
        t2i_adapter_conditioning_factor: float = 1.0,

        upscaler_model_path: Optional[str] = None,  # add latent
        upscaler_increases_size: float = 1.5,
        esrgan_tile: int = 100,
        esrgan_tile_overlap: int = 10,
        hires_steps: int = 25,
        hires_denoising_strength: float = 0.35,
        hires_prompt: str = "",
        hires_negative_prompt: str = "",
        hires_sampler: str = "Use same sampler",

        ip_adapter_image: Optional[Any] = [],  # str Image
        ip_adapter_mask: Optional[Any] = [],  # str Image
        ip_adapter_model: Optional[Any] = [],  # str
        ip_adapter_scale: Optional[Any] = [1.0],  # float
        ip_adapter_mode: Optional[Any] = ["original"],  # str: original, style, layout, style+layout

        loop_generation: int = 1,
        display_images: bool = False,
        save_generated_images: bool = True,
        image_storage_location: str = "./images",
        generator_in_cpu: bool = False,
        leave_progress_bar: bool = False,
        disable_progress_bar: bool = False,
        hires_before_adetailer: bool = False,
        hires_after_adetailer: bool = True,
        retain_compel_previous_load: bool = False,
        retain_detailfix_model_previous_load: bool = False,
        retain_hires_model_previous_load: bool = False,
        image_previews: bool = False,
        xformers_memory_efficient_attention: bool = False,
        gui_active: bool = False,
    ):

        """
        The call function for the generation.

        Args:
            prompt (str , optional):
                The prompt or prompts to guide image generation.
            negative_prompt (str , optional):
                The prompt or prompts to guide what to not include in image generation. Ignored when not using guidance (`guidance_scale < 1`).
            img_height (int, optional, defaults to 512):
                The height in pixels of the generated image.
            img_width (int, optional, defaults to 512):
                The width in pixels of the generated image.
            num_images (int, optional, defaults to 1):
                The number of images to generate per prompt.
            num_steps (int, optional, defaults to 30):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (float, optional, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            clip_skip (bool, optional):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. It can be placed on
                the penultimate (True) or last layer (False).
            seed (int, optional, defaults to -1):
                A seed for controlling the randomness of the image generation process. -1 design a random seed.
            sampler (str, optional, defaults to "DPM++ 2M"):
                The sampler used for the generation process.
                To see all the valid sampler names, use the following code:

                ```python
                from stablepy import scheduler_names
                print(scheduler_names)
                ```
            syntax_weights (str, optional, defaults to "Classic"):
                Specifies the type of syntax weights and emphasis used during generation.
                "Classic" is (word:weight), "Compel" is (word)weight.
                To see all the valid syntax weight options, use the following code:

                ```python
                from stablepy import ALL_PROMPT_WEIGHT_OPTIONS
                print(ALL_PROMPT_WEIGHT_OPTIONS)
                ```
            lora_A (str, optional):
                Placeholder for lora A parameter.
            lora_scale_A (float, optional, defaults to 1.0):
                Placeholder for lora scale A parameter.
            lora_B (str, optional):
                Placeholder for lora B parameter.
            lora_scale_B (float, optional, defaults to 1.0):
                Placeholder for lora scale B parameter.
            lora_C (str, optional):
                Placeholder for lora C parameter.
            lora_scale_C (float, optional, defaults to 1.0):
                Placeholder for lora scale C parameter.
            lora_D (str, optional):
                Placeholder for lora D parameter.
            lora_scale_D (float, optional, defaults to 1.0):
                Placeholder for lora scale D parameter.
            lora_E (str, optional):
                Placeholder for lora E parameter.
            lora_scale_E (float, optional, defaults to 1.0):
                Placeholder for lora scale E parameter.
            textual_inversion (List[Tuple[str, str]], optional, defaults to []):
                Placeholder for textual inversion list of tuples. Help the model to adapt to a particular
                style. [("<token_activation>","<path_embeding>"),...]
            FreeU (bool, optional, defaults to False):
                Is a method that substantially improves diffusion model sample quality at no costs.
            adetailer_A (bool, optional, defaults to False):
                Guided Inpainting to Correct Image, it is preferable to use low values for strength.
            adetailer_A_params (Dict[str, Any], optional, defaults to {}):
                Placeholder for adetailer_A parameters in a dict example {"prompt": "my prompt", "inpaint_only": True ...}.
                If not specified, default values will be used:
                - face_detector_ad (bool): Indicates whether face detection is enabled. Defaults to True.
                - person_detector_ad (bool): Indicates whether person detection is enabled. Defaults to True.
                - hand_detector_ad (bool): Indicates whether hand detection is enabled. Defaults to False.
                - prompt (str): A prompt for the adetailer_A. Defaults to an empty string.
                - negative_prompt (str): A negative prompt for the adetailer_A. Defaults to an empty string.
                - strength (float): The strength parameter value. Defaults to 0.35.
                - mask_dilation (int): The mask dilation value. Defaults to 4.
                - mask_blur (int): The mask blur value. Defaults to 4.
                - mask_padding (int): The mask padding value. Defaults to 32.
                - inpaint_only (bool): Indicates if only inpainting is to be performed. Defaults to True. False is img2img mode
                - sampler (str): The sampler type to be used. Defaults to "Use same sampler".
            adetailer_B (bool, optional, defaults to False):
                Guided Inpainting to Correct Image, it is preferable to use low values for strength.
            adetailer_B_params (Dict[str, Any], optional, defaults to {}):
                Placeholder for adetailer_B parameters in a dict example {"prompt": "my prompt", "inpaint_only": True ...}.
                If not specified, default values will be used.
            style_prompt (str, optional):
                If a style that is in STYLE_NAMES is specified, it will be added to the original prompt and negative prompt.
            style_json_file (str, optional):
                JSON with styles to be applied and used in style_prompt.
            pag_scale (float, optional):
                Perturbed Attention Guidance (PAG) enhances image generation quality without the need for training.
                If it is used, it is recommended to use values close to 3.0 for good results.
            upscaler_model_path (str, optional):
                This is the path of the ESRGAN model that will be used for the upscale; on the other hand,
                you can also use simply 'Lanczos', 'Nearest,' or 'Latent,' the latter of which has variants
                that can be consulted in the following code:

                ```python
                from stablepy import LATENT_UPSCALERS
                print(LATENT_UPSCALERS)
                ```
            upscaler_increases_size (float, optional, defaults to 1.5):
                Placeholder for upscaler increases size parameter.
            esrgan_tile (int, optional, defaults to 100):
                Tile if use a ESRGAN model.
            esrgan_tile_overlap (int, optional, defaults to 100):
                Tile overlap if use a ESRGAN model.
            hires_steps (int, optional, defaults to 25):
                The number of denoising steps for hires. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            hires_denoising_strength (float, optional, defaults to 0.35):
                Strength parameter for the hires.
            hires_prompt (str , optional):
                The prompt for hires. If not specified, the main prompt will be used.
            hires_negative_prompt (str , optional):
                The negative prompt for hires. If not specified, the main negative prompt will be used.
            hires_sampler (str, optional, defaults to "Use same sampler"):
                The sampler used for the hires generation process. If not specified, the main sampler will be used.
            image (Any, optional):
                The image to be used for the Inpaint, ControlNet, or T2I adapter.
            preprocessor_name (str, optional, defaults to "None"):
                Preprocessor name for ControlNet.
            preprocess_resolution (int, optional, defaults to 512):
                Preprocess resolution for the Inpaint, ControlNet, or T2I adapter.
            image_resolution (int, optional, defaults to 512):
                Image resolution for the Img2Img, Inpaint, ControlNet, or T2I adapter.
            image_mask (Any, optional):
                Path image mask for the Inpaint.
            strength (float, optional, defaults to 0.35):
                Strength parameter for the Inpaint and Img2Img.
            low_threshold (int, optional, defaults to 100):
                Low threshold parameter for ControlNet and T2I Adapter Canny.
            high_threshold (int, optional, defaults to 200):
                High threshold parameter for ControlNet and T2I Adapter Canny.
            value_threshold (float, optional, defaults to 0.1):
                Value threshold parameter for ControlNet MLSD.
            distance_threshold (float, optional, defaults to 0.1):
                Distance threshold parameter for ControlNet MLSD.
            controlnet_conditioning_scale (float, optional, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`. Used in ControlNet and Inpaint
            control_guidance_start (float, optional, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying. Used in ControlNet and Inpaint
            control_guidance_end (float, optional, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying. Used in ControlNet and Inpaint
            t2i_adapter_preprocessor (bool, optional, defaults to True):
                Preprocessor for the image in sdxl_canny by default is True.
            t2i_adapter_conditioning_scale (float, optional, defaults to 1.0):
                The outputs of the adapter are multiplied by `t2i_adapter_conditioning_scale` before they are added to the
                residual in the original unet.
            t2i_adapter_conditioning_factor (float, optional, defaults to 1.0):
                The fraction of timesteps for which adapter should be applied. If `t2i_adapter_conditioning_factor` is
                `0.0`, adapter is not applied at all. If `t2i_adapter_conditioning_factor` is `1.0`, adapter is applied for
                all timesteps. If `t2i_adapter_conditioning_factor` is `0.5`, adapter is applied for half of the timesteps.
            loop_generation (int, optional, defaults to 1):
                The number of times the specified `num_images` will be generated.
            display_images (bool, optional, defaults to False):
                If you use a notebook, you will be able to display the images generated with this parameter.
            save_generated_images (bool, optional, defaults to True):
                By default, the generated images are saved in the current location within the 'images' folder. You can disable this with this parameter.
            image_storage_location (str , optional, defaults to "./images"):
                The directory where the generated images are saved.
            generator_in_cpu (bool, optional, defaults to False):
                The generator by default is specified on the GPU. To obtain more consistent results across various environments,
                it is preferable to use the generator on the CPU.
            leave_progress_bar (bool, optional, defaults to False):
                Leave the progress bar after generating the image.
            disable_progress_bar (bool, optional, defaults to False):
                Do not display the progress bar during image generation.
            hires_before_adetailer (bool, optional, defaults to False):
                Apply an upscale and high-resolution fix before adetailer.
            hires_after_adetailer (bool, optional, defaults to True):
                Apply an upscale and high-resolution fix after adetailer.
            retain_compel_previous_load (bool, optional, defaults to False):
                The previous compel remains preloaded in memory.
            retain_detailfix_model_previous_load (bool, optional, defaults to False):
                The previous adetailer model remains preloaded in memory.
            retain_hires_model_previous_load (bool, optional, defaults to False):
                The previous hires model remains preloaded in memory.
            ip_adapter_image (Optional[Any], optional, default=[]):
                Image path or list of image paths for ip adapter.
            ip_adapter_mask (Optional[Any], optional, default=[]):
                Mask image path or list of mask image paths for ip adapter.
            ip_adapter_model (Optional[Any], optional, default=[]):
                Adapter model name or list of adapter model names.

                To see all the valid model names for SD1.5:
                ```python
                from stablepy import IP_ADAPTERS_SD
                print(IP_ADAPTERS_SD)
                ```

                To see all the valid model names for SDXL:
                ```python
                from stablepy import IP_ADAPTERS_SDXL
                print(IP_ADAPTERS_SDXL)
                ```
            ip_adapter_scale (Optional[Any], optional, default=[1.0]):
                Scaling factor or list of scaling factors for the ip adapter models.
            ip_adapter_mode (Optional[Any], optional, default=['original']):
                Adapter mode or list of adapter modes. Possible values are 'original', 'style', 'layout', 'style+layout'.
            image_previews (bool, optional, defaults to False):
                Displaying the image denoising process.
            xformers_memory_efficient_attention (bool, optional, defaults to False):
                Improves generation time, currently disabled.
            gui_active (bool, optional, defaults to False):
                utility when used with a GUI, it changes the behavior especially by displaying confirmation messages or options.

        Specific parameter usage details:

            Additional parameters that will be used in Inpaint:
                - image
                - image_mask
                - image_resolution
                - strength
                for SD 1.5:
                  - controlnet_conditioning_scale
                  - control_guidance_start
                  - control_guidance_end

            Additional parameters that will be used in img2img:
                - image
                - image_resolution
                - strength

            Additional parameters that will be used in ControlNet for SD 1.5 and SDXL depending on the task:
                - image
                - preprocessor_name
                - preprocess_resolution
                - image_resolution
                - controlnet_conditioning_scale
                - control_guidance_start
                - control_guidance_end
                for Canny:
                    - low_threshold
                    - high_threshold
                for MLSD:
                    - value_threshold
                    - distance_threshold

            Additional parameters that will be used in T2I adapter for SDXL depending on the task:
                - image
                - preprocess_resolution
                - image_resolution
                - t2i_adapter_preprocessor
                - t2i_adapter_conditioning_scale
                - t2i_adapter_conditioning_factor

        """

        if not seed:
            seed = -1
        if self.task_name != "txt2img" and image is None:
            raise ValueError(
                "You need to specify the <image> for this task."
            )
        if hires_steps < 2 and upscaler_model_path in LATENT_UPSCALERS:
            raise ValueError("Latent upscaler requires hires_steps. Use at least 2 steps.")
        if img_height % 8 != 0:
            img_height = img_height + (8 - img_height % 8)
            logger.warning(f"Height must be divisible by 8, changed to {str(img_height)}")
        if img_width % 8 != 0:
            img_width = img_width + (8 - img_width % 8)
            logger.warning(f"Width must be divisible by 8, changed to {str(img_width)}")
        if image_resolution % 8 != 0:
            image_resolution = image_resolution + (8 - image_resolution % 8)
            logger.warning(f"Image resolution must be divisible by 8, changed to {str(image_resolution)}")
        if control_guidance_start >= control_guidance_end:
            logger.error(
                "Control guidance start (ControlNet Start Threshold) "
                "cannot be larger or equal to control guidance end ("
                "ControlNet Stop Threshold). The default values 0.0 and "
                "1.0 will be used."
            )
            control_guidance_start, control_guidance_end = 0.0, 1.0

        if (ip_adapter_image and not ip_adapter_model) or (not ip_adapter_image and ip_adapter_model):
            raise ValueError(
                "Ip adapter require the ip adapter image and the ip adapter model for the task"
            )

        self.gui_active = gui_active
        self.image_previews = image_previews

        if self.pipe is None:
            self.load_pipe(
                self.base_model_id,
                task_name=self.task_name,
                vae_model=self.vae_model,
                reload=True,
            )

        pag_scale_is_true = bool(pag_scale)
        hasattr_pipe_pag = hasattr(self.pipe, "set_pag_applied_layers")
        if pag_scale_is_true != hasattr_pipe_pag:
            self.switch_pipe_class(
                self.class_name,
                self.task_name,
                self.model_id_task,
                enable_pag=bool(pag_scale),
            )

        self.pipe.set_progress_bar_config(leave=leave_progress_bar)
        self.pipe.set_progress_bar_config(disable=disable_progress_bar)

        xformers_memory_efficient_attention = False  # disabled
        if xformers_memory_efficient_attention and torch.cuda.is_available():
            self.pipe.disable_xformers_memory_efficient_attention()
        self.pipe.to(self.device)

        # Load style prompt file
        if style_json_file != "" and style_json_file != self.style_json_file:
            self.load_style_file(style_json_file)
        # Set style
        if isinstance(style_prompt, str):
            style_prompt = [style_prompt]
        if style_prompt != [""]:
            prompt, negative_prompt = apply_style(
                style_prompt, prompt, negative_prompt, self.styles_data, self.STYLE_NAMES
            )

        # LoRA load
        self.lora_status = self.lora_merge(
            lora_A, lora_scale_A,
            lora_B, lora_scale_B,
            lora_C, lora_scale_C,
            lora_D, lora_scale_D,
            lora_E, lora_scale_E,
        )

        if sampler in FLASH_AUTO_LOAD_SAMPLER and self.flash_config is None:
            # First load
            flash_task_lora = FLASH_LORA[self.class_name][sampler]
            self.process_lora(flash_task_lora, 1.0)
            self.flash_config = flash_task_lora
            logger.info(sampler)
        elif sampler not in FLASH_AUTO_LOAD_SAMPLER and self.flash_config is not None:
            # Unload
            self.process_lora(self.flash_config, 1.0, unload=True)
            self.flash_config = None
        elif self.flash_config is not None:
            flash_task_lora = FLASH_LORA[self.class_name][sampler]
            if flash_task_lora == self.flash_config:
                # Same
                pass
            else:
                # Change flash lora
                logger.debug(f"Unload '{self.flash_config}' and load '{flash_task_lora}'")
                self.process_lora(self.flash_config, 1.0, unload=True)
                self.process_lora(flash_task_lora, 1.0)
                self.flash_config = flash_task_lora
            logger.info(sampler)

        if not isinstance(ip_adapter_image, list):
            ip_adapter_image = [ip_adapter_image]
        if not isinstance(ip_adapter_mask, list):
            ip_adapter_mask = [ip_adapter_mask]
        if not isinstance(ip_adapter_model, list):
            ip_adapter_model = [ip_adapter_model]
        if not isinstance(ip_adapter_scale, list):
            ip_adapter_scale = [ip_adapter_scale]
        if not isinstance(ip_adapter_mode, list):
            ip_adapter_mode = [ip_adapter_mode]

        ip_weights = [IP_ADAPTER_MODELS[self.class_name][name] for name in ip_adapter_model]
        ip_scales = [float(s) for s in ip_adapter_scale]
        ip_masks = [
            [load_image(mask__) if mask__ else None for mask__ in sublist]
            if isinstance(sublist, list)
            else load_image(sublist) if sublist
            else None
            for sublist in ip_adapter_mask
        ]
        ip_images = [
            load_image(ip_img) if not isinstance(ip_img, list)
            else [load_image(img__) for img__ in ip_img]
            for ip_img in ip_adapter_image
        ]

        if self.ip_adapter_config is None and ip_adapter_image:
            # First load
            logger.info("Loading IP adapter")
            self.set_ip_adapter_model(ip_weights)

        elif self.ip_adapter_config is not None and not ip_adapter_image:
            # Unload
            logger.debug("IP adapter unload all")
            self.pipe.unload_ip_adapter()
            self.ip_adapter_config = None
        elif self.ip_adapter_config is not None:
            if self.ip_adapter_config == [data[2] for data in ip_weights]:
                logger.info("IP adapter")
            else:
                # change or retain same
                logger.debug("IP adapter reload all")
                self.pipe.unload_ip_adapter()
                self.ip_adapter_config = None
                logger.info("Loading IP adapter")
                self.set_ip_adapter_model(ip_weights)

        if self.ip_adapter_config:
            self.set_ip_adapter_multimode_scale(ip_scales, ip_adapter_mode)
            self.pipe.to(self.device)

        # FreeU
        if FreeU:
            logger.info("FreeU active")
            if self.class_name == "StableDiffusionPipeline":
                # sd
                self.pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
            else:
                # sdxl
                self.pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
            self.FreeU = True
        elif self.FreeU:
            self.pipe.disable_freeu()
            self.FreeU = False

        # Prompt Optimizations
        if hasattr(self, "compel") and not retain_compel_previous_load:
            del self.compel

        prompt_emb, negative_prompt_emb = self.create_prompt_embeds(
            prompt=prompt,
            negative_prompt=negative_prompt,
            textual_inversion=textual_inversion,
            clip_skip=clip_skip,
            syntax_weights=syntax_weights,
        )

        if self.class_name != "StableDiffusionPipeline":
            # Additional prompt for SDXL
            conditioning, pooled = prompt_emb.clone(), negative_prompt_emb.clone()
            prompt_emb = negative_prompt_emb = None

        if torch.cuda.is_available() and xformers_memory_efficient_attention:
            if xformers_memory_efficient_attention:
                self.pipe.enable_xformers_memory_efficient_attention()
            else:
                self.pipe.disable_xformers_memory_efficient_attention()

        try:
            # self.pipe.scheduler = DPMSolverSinglestepScheduler() # fix default params by random scheduler, not recomn
            self.pipe.scheduler = self.get_scheduler(sampler)
        except Exception as e:
            logger.debug(f"{e}")
            torch.cuda.empty_cache()
            gc.collect()
            raise RuntimeError("Error in sampler, please try again")

        self.pipe.safety_checker = None

        # Reference image
        if self.task_name != "txt2img":
            array_rgb = convert_image_to_numpy_array(image, gui_active)

        control_image = None

        # Run preprocess
        if self.task_name == "inpaint":
            # Get mask for Inpaint
            control_image, control_mask, tensor_control_image = self.process_inpaint(
                image=array_rgb,
                image_resolution=image_resolution,
                image_mask=image_mask,
                gui_active=gui_active,
            )

        elif self.task_name != "txt2img":
            control_image = self.get_image_preprocess(
                image=array_rgb,
                image_resolution=image_resolution,
                preprocess_resolution=preprocess_resolution,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                preprocessor_name=preprocessor_name,
                value_threshold=value_threshold,
                distance_threshold=distance_threshold,
                t2i_adapter_preprocessor=t2i_adapter_preprocessor,
                recolor_gamma_correction=recolor_gamma_correction,
            )

        # Task Parameters
        pipe_params_config = {
                "prompt": None,
                "negative_prompt": None,
                "num_inference_steps": num_steps,
                "guidance_scale": guidance_scale,
                "clip_skip": None,
                "num_images_per_prompt": num_images,
        }

        if hasattr(self.pipe, "set_pag_applied_layers"):
            pipe_params_config["pag_scale"] = float(pag_scale)

        if self.task_name == "txt2img":
            pipe_params_config["height"] = img_height
            pipe_params_config["width"] = img_width
        else:
            pipe_params_config["image"] = control_image
            logger.info(f"Image resolution: {str(control_image.size)}")

        if self.class_name == "StableDiffusionPipeline":
            pipe_params_config["prompt_embeds"] = prompt_emb
            pipe_params_config["negative_prompt_embeds"] = negative_prompt_emb

            if self.task_name == "inpaint":
                pipe_params_config["strength"] = strength
                pipe_params_config["mask_image"] = control_mask
                pipe_params_config["control_image"] = tensor_control_image
                pipe_params_config[
                    "controlnet_conditioning_scale"
                ] = float(controlnet_conditioning_scale)
                pipe_params_config["control_guidance_start"] = float(control_guidance_start)
                pipe_params_config["control_guidance_end"] = float(control_guidance_end)
                pipe_params_config["eta"] = 1.0
            elif self.task_name not in ["txt2img", "inpaint", "img2img"]:
                pipe_params_config[
                    "controlnet_conditioning_scale"
                ] = float(controlnet_conditioning_scale)
                pipe_params_config["control_guidance_start"] = float(control_guidance_start)
                pipe_params_config["control_guidance_end"] = float(control_guidance_end)
            elif self.task_name == "img2img":
                pipe_params_config["strength"] = strength
                pipe_params_config["eta"] = 1.0

        elif self.class_name == "StableDiffusionXLPipeline":
            pipe_params_config["prompt_embeds"] = conditioning[0:1]
            pipe_params_config["pooled_prompt_embeds"] = pooled[0:1]
            pipe_params_config["negative_prompt_embeds"] = conditioning[1:2]
            pipe_params_config["negative_pooled_prompt_embeds"] = pooled[1:2]

            if self.task_name == "inpaint":
                pipe_params_config["strength"] = strength
                pipe_params_config["mask_image"] = control_mask
                pipe_params_config["height"] = control_image.size[1]
                pipe_params_config["width"] = control_image.size[0]
            elif self.task_name not in ["txt2img", "inpaint", "img2img"]:
                if "t2i" not in self.task_name:
                    pipe_params_config[
                        "controlnet_conditioning_scale"
                    ] = float(controlnet_conditioning_scale)
                    pipe_params_config["control_guidance_start"] = float(control_guidance_start)
                    pipe_params_config["control_guidance_end"] = float(control_guidance_end)
                else:
                    pipe_params_config["adapter_conditioning_scale"] = float(t2i_adapter_conditioning_scale)
                    pipe_params_config["adapter_conditioning_factor"] = float(t2i_adapter_conditioning_factor)
            elif self.task_name == "img2img":
                pipe_params_config["strength"] = strength

        if self.ip_adapter_config:
            # maybe need cache embeds
            ip_adapter_embeds, ip_adapter_masks = self.get_ip_embeds(
                guidance_scale, ip_images, num_images, ip_masks
            )

            pipe_params_config["ip_adapter_image_embeds"] = ip_adapter_embeds
            if ip_adapter_masks:
                pipe_params_config["cross_attention_kwargs"] = {
                    "ip_adapter_masks": ip_adapter_masks
                }

        post_processing_params = dict(
            display_images=display_images
        )

        # detailfix params and pipe global
        if adetailer_A or adetailer_B:

            # global params detailfix
            default_params_detailfix = {
                "face_detector_ad": True,
                "person_detector_ad": True,
                "hand_detector_ad": False,
                "prompt": "",
                "negative_prompt": "",
                "strength": 0.35,
                "mask_dilation": 4,
                "mask_blur": 4,
                "mask_padding": 32,
                # "sampler": "Use same sampler",
                # "inpaint_only": True,
            }

            # Pipe detailfix_pipe
            if not hasattr(self, "detailfix_pipe") or not retain_detailfix_model_previous_load:
                if adetailer_A_params.get("inpaint_only", False) == True or adetailer_B_params.get("inpaint_only", False) == True:
                    detailfix_pipe = custom_task_model_loader(
                        pipe=self.pipe,
                        model_category="detailfix",
                        task_name=self.task_name,
                        torch_dtype=self.type_model_precision
                    )
                else:
                    detailfix_pipe = custom_task_model_loader(
                        pipe=self.pipe,
                        model_category="detailfix_img2img",
                        task_name=self.task_name,
                        torch_dtype=self.type_model_precision
                    )
                if hasattr(self, "detailfix_pipe"):
                    del self.detailfix_pipe
            if retain_detailfix_model_previous_load:
                if hasattr(self, "detailfix_pipe"):
                    detailfix_pipe = self.detailfix_pipe
                else:
                    self.detailfix_pipe = detailfix_pipe
            adetailer_A_params.pop("inpaint_only", None)
            adetailer_B_params.pop("inpaint_only", None)

            # Define base scheduler detailfix
            detailfix_pipe.default_scheduler = copy.deepcopy(self.default_scheduler)
            if adetailer_A_params.get("sampler", "Use same sampler") != "Use same sampler":
                logger.debug("detailfix_pipe will use the sampler from adetailer_A")
                detailfix_pipe.scheduler = self.get_scheduler(adetailer_A_params["sampler"])
            adetailer_A_params.pop("sampler", None)
            if adetailer_B_params.get("sampler", "Use same sampler") != "Use same sampler":
                logger.debug("detailfix_pipe will use the sampler from adetailer_B")
                detailfix_pipe.scheduler = self.get_scheduler(adetailer_A_params["sampler"])
            adetailer_B_params.pop("sampler", None)

            detailfix_pipe.set_progress_bar_config(leave=leave_progress_bar)
            detailfix_pipe.set_progress_bar_config(disable=disable_progress_bar)
            detailfix_pipe.to(self.device)
            torch.cuda.empty_cache()
            gc.collect()

            post_processing_params["detailfix_pipe"] = detailfix_pipe

        if adetailer_A:
            for key_param, default_value in default_params_detailfix.items():
                if key_param not in adetailer_A_params:
                    adetailer_A_params[key_param] = default_value
                elif type(default_value) != type(adetailer_A_params[key_param]):
                    logger.warning(f"DetailFix A: Error type param, set default {str(key_param)}")
                    adetailer_A_params[key_param] = default_value

            detailfix_params_A = {
                "prompt": adetailer_A_params["prompt"],
                "negative_prompt": adetailer_A_params["negative_prompt"],
                "strength": adetailer_A_params["strength"],
                "num_inference_steps": int(num_steps * 1.5),
                "guidance_scale": guidance_scale,
            }

            if self.ip_adapter_config:
                detailfix_params_A["ip_adapter_image_embeds"] = ip_adapter_embeds
                if ip_adapter_masks:
                    detailfix_params_A["cross_attention_kwargs"] = {
                        "ip_adapter_masks": ip_adapter_masks
                    }

            # clear params yolo
            adetailer_A_params.pop('strength', None)
            adetailer_A_params.pop('prompt', None)
            adetailer_A_params.pop('negative_prompt', None)

            # Verify prompt detailfix_params_A and get valid
            prompt_empty_detailfix_A, negative_prompt_empty_detailfix_A, prompt_df_A, negative_prompt_df_A = process_prompts_valid(
                detailfix_params_A["prompt"], detailfix_params_A["negative_prompt"], prompt, negative_prompt
            )

            # Params detailfix
            if self.class_name == "StableDiffusionPipeline":
                # SD detailfix
                # detailfix_params_A["controlnet_conditioning_scale"] = controlnet_conditioning_scale
                # detailfix_params_A["control_guidance_start"] = control_guidance_start
                # detailfix_params_A["control_guidance_end"] = control_guidance_end

                if prompt_empty_detailfix_A and negative_prompt_empty_detailfix_A:
                    detailfix_params_A["prompt_embeds"] = prompt_emb
                    detailfix_params_A["negative_prompt_embeds"] = negative_prompt_emb
                else:
                    prompt_emb_ad, negative_prompt_emb_ad = self.create_prompt_embeds(
                        prompt=prompt_df_A,
                        negative_prompt=negative_prompt_df_A,
                        textual_inversion=textual_inversion,
                        clip_skip=clip_skip,
                        syntax_weights=syntax_weights,
                    )
                    detailfix_params_A["prompt_embeds"] = prompt_emb_ad
                    detailfix_params_A["negative_prompt_embeds"] = negative_prompt_emb_ad

                detailfix_params_A["prompt"] = None
                detailfix_params_A["negative_prompt"] = None

            else:
                # SDXL detailfix
                if prompt_empty_detailfix_A and negative_prompt_empty_detailfix_A:
                    conditioning_detailfix_A, pooled_detailfix_A = conditioning, pooled
                else:
                    conditioning_detailfix_A, pooled_detailfix_A = self.create_prompt_embeds(
                        prompt=prompt_df_A,
                        negative_prompt=negative_prompt_df_A,
                        textual_inversion=textual_inversion,
                        clip_skip=clip_skip,
                        syntax_weights=syntax_weights,
                    )

                detailfix_params_A.pop('prompt', None)
                detailfix_params_A.pop('negative_prompt', None)
                detailfix_params_A["prompt_embeds"] = conditioning_detailfix_A[0:1]
                detailfix_params_A["pooled_prompt_embeds"] = pooled_detailfix_A[0:1]
                detailfix_params_A["negative_prompt_embeds"] = conditioning_detailfix_A[1:2]
                detailfix_params_A["negative_pooled_prompt_embeds"] = pooled_detailfix_A[1:2]

            logger.debug(f"detailfix A prompt empty {prompt_empty_detailfix_A, negative_prompt_empty_detailfix_A}")
            if not prompt_empty_detailfix_A or not negative_prompt_empty_detailfix_A:
                logger.debug(f"Prompts detailfix A {prompt_df_A, negative_prompt_df_A}")
            logger.debug(f"Pipe params detailfix A \n{detailfix_params_A}")
            logger.debug(f"Params detailfix A \n{adetailer_A_params}")

            post_processing_params["detailfix_params_A"] = detailfix_params_A

        if adetailer_B:
            for key_param, default_value in default_params_detailfix.items():
                if key_param not in adetailer_B_params:
                    adetailer_B_params[key_param] = default_value
                elif type(default_value) != type(adetailer_B_params[key_param]):
                    logger.warning(f"DetailfFix B: Error type param, set default {str(key_param)}")
                    adetailer_B_params[key_param] = default_value

            detailfix_params_B = {
                "prompt": adetailer_B_params["prompt"],
                "negative_prompt": adetailer_B_params["negative_prompt"],
                "strength": adetailer_B_params["strength"],
                "num_inference_steps": int(num_steps * 1.5),
                "guidance_scale": guidance_scale,
            }

            if self.ip_adapter_config:
                detailfix_params_B["ip_adapter_image_embeds"] = ip_adapter_embeds
                if ip_adapter_masks:
                    detailfix_params_B["cross_attention_kwargs"] = {
                        "ip_adapter_masks": ip_adapter_masks
                    }

            # clear params yolo
            adetailer_B_params.pop('strength', None)
            adetailer_B_params.pop('prompt', None)
            adetailer_B_params.pop('negative_prompt', None)

            # Verify prompt detailfix_params_B and get valid
            prompt_empty_detailfix_B, negative_prompt_empty_detailfix_B, prompt_df_B, negative_prompt_df_B = process_prompts_valid(
                detailfix_params_B["prompt"], detailfix_params_B["negative_prompt"], prompt, negative_prompt
            )

            # Params detailfix
            if self.class_name == "StableDiffusionPipeline":
                # SD detailfix
                # detailfix_params_B["controlnet_conditioning_scale"] = controlnet_conditioning_scale
                # detailfix_params_B["control_guidance_start"] = control_guidance_start
                # detailfix_params_B["control_guidance_end"] = control_guidance_end

                if prompt_empty_detailfix_B and negative_prompt_empty_detailfix_B:
                    detailfix_params_B["prompt_embeds"] = prompt_emb
                    detailfix_params_B["negative_prompt_embeds"] = negative_prompt_emb
                else:
                    prompt_emb_ad_b, negative_prompt_emb_ad_b = self.create_prompt_embeds(
                        prompt=prompt_df_B,
                        negative_prompt=negative_prompt_df_B,
                        textual_inversion=textual_inversion,
                        clip_skip=clip_skip,
                        syntax_weights=syntax_weights,
                    )
                    detailfix_params_B["prompt_embeds"] = prompt_emb_ad_b
                    detailfix_params_B["negative_prompt_embeds"] = negative_prompt_emb_ad_b
                detailfix_params_B["prompt"] = None
                detailfix_params_B["negative_prompt"] = None
            else:
                # SDXL detailfix
                if prompt_empty_detailfix_B and negative_prompt_empty_detailfix_B:
                    conditioning_detailfix_B, pooled_detailfix_B = conditioning, pooled
                else:
                    conditioning_detailfix_B, pooled_detailfix_B = self.create_prompt_embeds(
                        prompt=prompt_df_B,
                        negative_prompt=negative_prompt_df_B,
                        textual_inversion=textual_inversion,
                        clip_skip=clip_skip,
                        syntax_weights=syntax_weights,
                    )
                detailfix_params_B.pop('prompt', None)
                detailfix_params_B.pop('negative_prompt', None)
                detailfix_params_B["prompt_embeds"] = conditioning_detailfix_B[0:1]
                detailfix_params_B["pooled_prompt_embeds"] = pooled_detailfix_B[0:1]
                detailfix_params_B["negative_prompt_embeds"] = conditioning_detailfix_B[1:2]
                detailfix_params_B["negative_pooled_prompt_embeds"] = pooled_detailfix_B[1:2]

            logger.debug(f"detailfix B prompt empty {prompt_empty_detailfix_B, negative_prompt_empty_detailfix_B}")
            if not prompt_empty_detailfix_B or not negative_prompt_empty_detailfix_B:
                logger.debug(f"Prompts detailfix B {prompt_df_B, negative_prompt_df_B}")
            logger.debug(f"Pipe params detailfix B \n{detailfix_params_B}")
            logger.debug(f"Params detailfix B \n{adetailer_B_params}")

            post_processing_params["detailfix_params_B"] = detailfix_params_B

        if hires_steps > 1 and upscaler_model_path is not None:
            # Hires params BASE
            hires_params_config = {
                "prompt": None,
                "negative_prompt": None,
                "num_inference_steps": hires_steps,
                "guidance_scale": guidance_scale,
                "clip_skip": None,
                "strength": hires_denoising_strength,
            }
            if self.class_name == "StableDiffusionPipeline":
                hires_params_config["eta"] = 1.0

            if self.ip_adapter_config:
                hires_params_config["ip_adapter_image_embeds"] = ip_adapter_embeds
                if ip_adapter_masks:
                    hires_params_config["cross_attention_kwargs"] = {
                        "ip_adapter_masks": ip_adapter_masks
                    }

            # Verify prompt hires and get valid
            hires_prompt_empty, hires_negative_prompt_empty, prompt_hires_valid, negative_prompt_hires_valid = process_prompts_valid(
                hires_prompt, hires_negative_prompt, prompt, negative_prompt
            )

            # Hires embed params
            if self.class_name == "StableDiffusionPipeline":
                if hires_prompt_empty and hires_negative_prompt_empty:
                    hires_params_config["prompt_embeds"] = prompt_emb
                    hires_params_config["negative_prompt_embeds"] = negative_prompt_emb
                else:
                    prompt_emb_hires, negative_prompt_emb_hires = self.create_prompt_embeds(
                        prompt=prompt_hires_valid,
                        negative_prompt=negative_prompt_hires_valid,
                        textual_inversion=textual_inversion,
                        clip_skip=clip_skip,
                        syntax_weights=syntax_weights,
                    )

                    hires_params_config["prompt_embeds"] = prompt_emb_hires
                    hires_params_config["negative_prompt_embeds"] = negative_prompt_emb_hires
            else:
                if hires_prompt_empty and hires_negative_prompt_empty:
                    hires_conditioning, hires_pooled = conditioning, pooled
                else:
                    hires_conditioning, hires_pooled = self.create_prompt_embeds(
                        prompt=prompt_hires_valid,
                        negative_prompt=negative_prompt_hires_valid,
                        textual_inversion=textual_inversion,
                        clip_skip=clip_skip,
                        syntax_weights=syntax_weights,
                    )

                hires_params_config.pop('prompt', None)
                hires_params_config.pop('negative_prompt', None)
                hires_params_config["prompt_embeds"] = hires_conditioning[0:1]
                hires_params_config["pooled_prompt_embeds"] = hires_pooled[0:1]
                hires_params_config["negative_prompt_embeds"] = hires_conditioning[1:2]
                hires_params_config["negative_pooled_prompt_embeds"] = hires_pooled[1:2]

            # Hires pipe
            if not hasattr(self, "hires_pipe") or not retain_hires_model_previous_load:
                hires_pipe = custom_task_model_loader(
                    pipe=self.pipe,
                    model_category="hires",
                    task_name=self.task_name,
                    torch_dtype=self.type_model_precision
                )
                if hasattr(self, "hires_pipe"):
                    del self.hires_pipe
            if retain_hires_model_previous_load:
                if hasattr(self, "hires_pipe"):
                    hires_pipe = self.hires_pipe
                else:
                    self.hires_pipe = hires_pipe

            # Hires scheduler
            if hires_sampler != "Use same sampler":
                logger.debug("New hires sampler")
                hires_pipe.scheduler = self.get_scheduler(hires_sampler)

            hires_pipe.set_progress_bar_config(leave=leave_progress_bar)
            hires_pipe.set_progress_bar_config(disable=disable_progress_bar)
            hires_pipe.to(self.device)
            torch.cuda.empty_cache()
            gc.collect()

            if (
                upscaler_model_path in LATENT_UPSCALERS
                and ((not adetailer_A and not adetailer_B) or hires_before_adetailer)
            ):
                pipe_params_config["output_type"] = "latent"

        else:
            hires_params_config = {}
            hires_pipe = None

        # Debug info
        try:
            logger.debug(f"INFO PIPE: {self.pipe.__class__.__name__}")
            logger.debug(f"text_encoder_type: {self.pipe.text_encoder.dtype}")
            logger.debug(f"unet_type: {self.pipe.unet.dtype}")
            logger.debug(f"vae_type: {self.pipe.vae.dtype}")
            logger.debug(f"pipe_type: {self.pipe.dtype}")
            logger.debug(f"scheduler_main_pipe: {self.pipe.scheduler}")
            if adetailer_A or adetailer_B:
                logger.debug(f"scheduler_detailfix: {detailfix_pipe.scheduler}")
            if hires_steps > 1 and upscaler_model_path is not None:
                logger.debug(f"scheduler_hires: {hires_pipe.scheduler}")
        except Exception as e:
            logger.debug(f"{str(e)}")

        metadata = [
            prompt,
            negative_prompt,
            self.base_model_id,
            self.vae_model,
            num_steps,
            guidance_scale,
            sampler,
            0000000000,  # calculate_seed,
            img_width,
            img_height,
            clip_skip,
        ]

        # === RUN PIPE === #
        handle_task = self.start_work if not image_previews else self.start_stream

        return handle_task(
            num_images,
            seed,
            adetailer_A,
            adetailer_A_params,
            adetailer_B,
            adetailer_B_params,
            upscaler_model_path,
            upscaler_increases_size,
            esrgan_tile,
            esrgan_tile_overlap,
            hires_steps,
            loop_generation,
            display_images,
            save_generated_images,
            image_storage_location,
            generator_in_cpu,
            hires_before_adetailer,
            hires_after_adetailer,
            retain_compel_previous_load,
            control_image,
            pipe_params_config,
            post_processing_params,
            hires_params_config,
            hires_pipe,
            metadata,
        )

    def post_processing(
        self,
        adetailer_A,
        adetailer_A_params,
        adetailer_B,
        adetailer_B_params,
        upscaler_model_path,
        upscaler_increases_size,
        esrgan_tile,
        esrgan_tile_overlap,
        hires_steps,
        loop_generation,
        display_images,
        save_generated_images,
        image_storage_location,
        hires_before_adetailer,
        hires_after_adetailer,
        control_image,
        post_processing_params,
        hires_params_config,
        hires_pipe,
        seeds,
        generators,
        images,
        metadata,
    ):
        if isinstance(images, torch.Tensor):
            images = [tl.unsqueeze(0) for tl in torch.unbind(images, dim=0)]

        if self.task_name not in ["txt2img", "inpaint", "img2img"]:
            images = [control_image] + images

        torch.cuda.empty_cache()
        gc.collect()

        if hires_before_adetailer and upscaler_model_path is not None:
            logger.debug(
                    "Hires before; same seed for each image (no batch)"
                )
            images = process_images_high_resolution(
                    images,
                    upscaler_model_path,
                    upscaler_increases_size,
                    esrgan_tile, esrgan_tile_overlap,
                    hires_steps, hires_params_config,
                    self.task_name,
                    generators[0],  # pipe_params_config["generator"][0], # no generator
                    hires_pipe,
                )

            # Adetailer stuff
        if adetailer_A or adetailer_B:
            # image_pil_list = []
            # for img_single in images:
            # image_ad = img_single.convert("RGB")
            # image_pil_list.append(image_ad)
            if self.task_name not in ["txt2img", "inpaint", "img2img"]:
                images = images[1:]

            if adetailer_A:
                images = ad_model_process(
                        pipe_params_df=post_processing_params["detailfix_params_A"],
                        detailfix_pipe=post_processing_params["detailfix_pipe"],
                        image_list_task=images,
                        **adetailer_A_params,
                    )
            if adetailer_B:
                images = ad_model_process(
                        pipe_params_df=post_processing_params["detailfix_params_B"],
                        detailfix_pipe=post_processing_params["detailfix_pipe"],
                        image_list_task=images,
                        **adetailer_B_params,
                    )

            if self.task_name not in ["txt2img", "inpaint", "img2img"]:
                images = [control_image] + images
                # del detailfix_pipe
            torch.cuda.empty_cache()
            gc.collect()

        if hires_after_adetailer and upscaler_model_path is not None:
            logger.debug(
                    "Hires after; same seed for each image (no batch)"
                )
            images = process_images_high_resolution(
                    images,
                    upscaler_model_path,
                    upscaler_increases_size,
                    esrgan_tile, esrgan_tile_overlap,
                    hires_steps, hires_params_config,
                    self.task_name,
                    generators[0],  # pipe_params_config["generator"][0], # no generator
                    hires_pipe,
                )

        logger.info(f"Seeds: {seeds}")

        # Show images if loop
        if display_images:
            mediapy.show_images(images)
            # logger.info(image_list)
            # del images
            if loop_generation > 1:
                time.sleep(0.5)

            # List images and save
        image_list = []
        image_metadata = []

        valid_seeds = [0] + seeds if self.task_name not in ["txt2img", "inpaint", "img2img"] else seeds
        for image_, seed_ in zip(images, valid_seeds):

            metadata[7] = seed_
            image_generation_data = get_string_metadata(metadata)

            image_path = "not saved in storage"
            if save_generated_images:
                image_path = save_pil_image_with_metadata(
                    image_, image_storage_location, image_generation_data
                )

            image_list.append(image_path)
            image_metadata.append(image_generation_data)

        torch.cuda.empty_cache()
        gc.collect()

        if image_list[0] != "not saved in storage":
            logger.info(image_list)

        return images, image_list, image_metadata

    def start_work(
        self,
        num_images,
        seed,
        adetailer_A,
        adetailer_A_params,
        adetailer_B,
        adetailer_B_params,
        upscaler_model_path,
        upscaler_increases_size,
        esrgan_tile,
        esrgan_tile_overlap,
        hires_steps,
        loop_generation,
        display_images,
        save_generated_images,
        image_storage_location,
        generator_in_cpu,
        hires_before_adetailer,
        hires_after_adetailer,
        retain_compel_previous_load,
        control_image,
        pipe_params_config,
        post_processing_params,
        hires_params_config,
        hires_pipe,
        metadata,
    ):
        for i in range(loop_generation):
            # number seed
            if seed == -1:
                seeds = [random.randint(0, 2147483647) for _ in range(num_images)]
            else:
                if num_images == 1:
                    seeds = [seed]
                else:
                    seeds = [seed] + [random.randint(0, 2147483647) for _ in range(num_images-1)]

            # generators
            generators = []  # List to store all the generators
            for calculate_seed in seeds:
                if generator_in_cpu or self.device.type == "cpu":
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
            pipe_params_config["generator"] = generators if self.task_name != "img2img" else generators[0]  # no list
            seeds = seeds if self.task_name != "img2img" else [seeds[0]] * num_images

            try:
                images = self.pipe(
                    **pipe_params_config,
                ).images

            except Exception as e:
                e = str(e)
                if "Tensor with 2 elements cannot be converted to Scalar" in e:
                    logger.debug(e)
                    logger.error("Error in sampler; trying with DDIM sampler")
                    self.pipe.scheduler = self.default_scheduler
                    self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
                    images = self.pipe(
                        **pipe_params_config,
                    ).images

                elif "The size of tensor a (0) must match the size of tensor b (3) at non-singleton" in e:
                    raise ValueError(
                        "steps / strength too low for the model to produce a satisfactory response"
                    )

                else:
                    raise ValueError(e)

            images, image_list, image_metadata = self.post_processing(
                adetailer_A,
                adetailer_A_params,
                adetailer_B,
                adetailer_B_params,
                upscaler_model_path,
                upscaler_increases_size,
                esrgan_tile,
                esrgan_tile_overlap,
                hires_steps,
                loop_generation,
                display_images,
                save_generated_images,
                image_storage_location,
                hires_before_adetailer,
                hires_after_adetailer,
                control_image,
                post_processing_params,
                hires_params_config,
                hires_pipe,
                seeds,
                generators,
                images,
                metadata,
            )

        if hasattr(self, "compel") and not retain_compel_previous_load:
            del self.compel
        torch.cuda.empty_cache()
        gc.collect()
        return images, [seeds, image_list, image_metadata]

    def start_stream(
        self,
        num_images,
        seed,
        adetailer_A,
        adetailer_A_params,
        adetailer_B,
        adetailer_B_params,
        upscaler_model_path,
        upscaler_increases_size,
        esrgan_tile,
        esrgan_tile_overlap,
        hires_steps,
        loop_generation,
        display_images,
        save_generated_images,
        image_storage_location,
        generator_in_cpu,
        hires_before_adetailer,
        hires_after_adetailer,
        retain_compel_previous_load,
        control_image,
        pipe_params_config,
        post_processing_params,
        hires_params_config,
        hires_pipe,
        metadata,
    ):
        for i in range(loop_generation):
            # number seed
            if seed == -1:
                seeds = [random.randint(0, 2147483647) for _ in range(num_images)]
            else:
                if num_images == 1:
                    seeds = [seed]
                else:
                    seeds = [seed] + [random.randint(0, 2147483647) for _ in range(num_images-1)]

            # generators
            generators = []  # List to store all the generators
            for calculate_seed in seeds:
                if generator_in_cpu or self.device.type == "cpu":
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
            pipe_params_config["generator"] = generators if self.task_name != "img2img" else generators[0]  # no list
            seeds = seeds if self.task_name != "img2img" else [seeds[0]] * num_images

            try:
                logger.debug("Start stream")
                # self.stream_config(5)
                stream = self.stream_preview(pipe_params_config)
                for img in stream:
                    if not isinstance(img, list):
                        img = [img]
                    yield img, [seeds, None, None]

                images = self.final_image

            except Exception as e:
                e = str(e)
                if "Tensor with 2 elements cannot be converted to Scalar" in e:
                    logger.debug(e)
                    logger.error("Error in sampler; trying with DDIM sampler")
                    self.pipe.scheduler = self.default_scheduler
                    self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
                    stream = self.stream_preview(pipe_params_config)
                    for img in stream:
                        if not isinstance(img, list):
                            img = [img]
                        yield img, [seeds, None, None]

                    images = self.final_image

                elif "The size of tensor a (0) must match the size of tensor b (3) at non-singleton" in e:
                    raise ValueError(
                        "steps / strength too low for the model to produce a satisfactory response"
                    )

                else:
                    raise ValueError(e)

            images, image_list, image_metadata = self.post_processing(
                adetailer_A,
                adetailer_A_params,
                adetailer_B,
                adetailer_B_params,
                upscaler_model_path,
                upscaler_increases_size,
                esrgan_tile,
                esrgan_tile_overlap,
                hires_steps,
                loop_generation,
                display_images,
                save_generated_images,
                image_storage_location,
                hires_before_adetailer,
                hires_after_adetailer,
                control_image,
                post_processing_params,
                hires_params_config,
                hires_pipe,
                seeds,
                generators,
                images,
                metadata,
            )

        if hasattr(self, "compel") and not retain_compel_previous_load:
            del self.compel
        torch.cuda.empty_cache()
        gc.collect()

        yield images, [seeds, image_list, image_metadata]
