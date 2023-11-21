import gc, time
import numpy as np
import PIL.Image
from diffusers import (
    ControlNetModel,
    DiffusionPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionPipeline,
    AutoencoderKL,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLAdapterPipeline,
    T2IAdapter,
    StableDiffusionXLPipeline,
    AutoPipelineForImage2Image
)
from huggingface_hub import hf_hub_download
import torch, random, json
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
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2DiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    DEISMultistepScheduler,
    UniPCMultistepScheduler,
    LCMScheduler,
    PNDMScheduler,
    KDPM2AncestralDiscreteScheduler,
)
from .prompt_weights import get_embed_new, add_comma_after_pattern_ti
from .utils import save_pil_image_with_metadata
from .lora_loader import lora_mix_load
from .inpainting_canvas import draw, make_inpaint_condition
from .adetailer import ad_model_process
from ..upscalers.esrgan import UpscalerESRGAN, UpscalerLanczos, UpscalerNearest
from ..logging.logging_setup import logger
from .extra_model_loaders import custom_task_model_loader
from .high_resolution import process_images_high_resolution
from .style_prompt_config import styles_data, STYLE_NAMES, get_json_content, apply_style
import os
from compel import Compel, ReturnedEmbeddingsType
import ipywidgets as widgets, mediapy
from IPython.display import display
from PIL import Image
from typing import Union, Optional, List, Tuple, Dict, Any, Callable
import logging, diffusers, copy, warnings
logging.getLogger("diffusers").setLevel(logging.ERROR)
#logging.getLogger("transformers").setLevel(logging.ERROR)
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

CONTROLNET_MODEL_IDS = {
    "openpose": "lllyasviel/control_v11p_sd15_openpose",
    "canny": "lllyasviel/control_v11p_sd15_canny",
    "mlsd": "lllyasviel/control_v11p_sd15_mlsd",
    "scribble": "lllyasviel/control_v11p_sd15_scribble",
    "softedge": "lllyasviel/control_v11p_sd15_softedge",
    "segmentation": "lllyasviel/control_v11p_sd15_seg",
    "depth": "lllyasviel/control_v11f1p_sd15_depth",
    "normalbae": "lllyasviel/control_v11p_sd15_normalbae",
    "lineart": "lllyasviel/control_v11p_sd15_lineart",
    "lineart_anime": "lllyasviel/control_v11p_sd15s2_lineart_anime",
    "shuffle": "lllyasviel/control_v11e_sd15_shuffle",
    "ip2p": "lllyasviel/control_v11e_sd15_ip2p",
    "inpaint": "lllyasviel/control_v11p_sd15_inpaint",
    "txt2img": "Nothinghere",
    "sdxl_canny": "TencentARC/t2i-adapter-canny-sdxl-1.0",
    "sdxl_sketch": "TencentARC/t2i-adapter-sketch-sdxl-1.0",
    "sdxl_lineart": "TencentARC/t2i-adapter-lineart-sdxl-1.0",
    "sdxl_depth-midas": "TencentARC/t2i-adapter-depth-midas-sdxl-1.0",
    "sdxl_openpose": "TencentARC/t2i-adapter-openpose-sdxl-1.0",
    #"sdxl_depth-zoe": "TencentARC/t2i-adapter-depth-zoe-sdxl-1.0",
    #"sdxl_recolor": "TencentARC/t2i-adapter-recolor-sdxl-1.0",
    "img2img": "Nothinghere",
}


# def download_all_controlnet_weights() -> None:
#     for model_id in CONTROLNET_MODEL_IDS.values():
#         ControlNetModel.from_pretrained(model_id)


SCHEDULER_CONFIG_MAP = {
    "DPM++ 2M": (DPMSolverMultistepScheduler, {}),
    "DPM++ 2M Karras": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True}),
    "DPM++ 2M SDE": (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++"}),
    "DPM++ 2M SDE Karras": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True, "algorithm_type": "sde-dpmsolver++"}),
    "DPM++ SDE": (DPMSolverSinglestepScheduler, {}),
    "DPM++ SDE Karras": (DPMSolverSinglestepScheduler, {"use_karras_sigmas": True}),
    "DPM2": (KDPM2DiscreteScheduler, {}),
    "DPM2 Karras": (KDPM2DiscreteScheduler, {"use_karras_sigmas": True}),
    "DPM2 a" : (KDPM2AncestralDiscreteScheduler, {}),
    "DPM2 a Karras" : (KDPM2AncestralDiscreteScheduler, {"use_karras_sigmas": True}),
    "Euler": (EulerDiscreteScheduler, {}),
    "Euler a": (EulerAncestralDiscreteScheduler, {}),
    "Heun": (HeunDiscreteScheduler, {}),
    "LMS": (LMSDiscreteScheduler, {}),
    "LMS Karras": (LMSDiscreteScheduler, {"use_karras_sigmas": True}),
    "DDIM": (DDIMScheduler, {}),
    "DEIS": (DEISMultistepScheduler, {}),
    "UniPC": (UniPCMultistepScheduler, {}),
    "PNDM" : (PNDMScheduler, {}),

    "DPM++ 2M Lu": (DPMSolverMultistepScheduler, {"use_lu_lambdas": True}),
    "DPM++ 2M Ef": (DPMSolverMultistepScheduler, {"euler_at_final": True}),
    "DPM++ 2M SDE Lu": (DPMSolverMultistepScheduler, {"use_lu_lambdas": True, "algorithm_type": "sde-dpmsolver++"}),
    "DPM++ 2M SDE Ef": (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++", "euler_at_final": True}),

    "LCM" : (LCMScheduler, {}),
}

scheduler_names = list(SCHEDULER_CONFIG_MAP.keys())

def process_prompts_valid(specific_prompt, specific_negative_prompt, prompt, negative_prompt):
    specific_prompt_empty = (specific_prompt in [None, ""])
    specific_negative_prompt_empty = (specific_negative_prompt in [None, ""])

    prompt_valid = prompt if specific_prompt_empty else specific_prompt
    negative_prompt_valid = negative_prompt if specific_negative_prompt_empty else specific_negative_prompt

    return specific_prompt_empty, specific_negative_prompt_empty, prompt_valid, negative_prompt_valid

class Model_Diffusers:
    def __init__(
        self,
        base_model_id: str = "runwayml/stable-diffusion-v1-5",
        task_name: str = "txt2img",
        vae_model=None,
        type_model_precision=torch.float16,
        sdxl_safetensors = False,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.base_model_id = ""
        self.task_name = ""
        self.vae_model = None
        self.type_model_precision = (
            type_model_precision if torch.cuda.is_available() else torch.float32
        )  # For SD 1.5

        self.load_pipe(
            base_model_id, task_name, vae_model, type_model_precision, sdxl_safetensors = sdxl_safetensors
        )
        self.preprocessor = Preprocessor()

        self.styles_data = styles_data
        self.STYLE_NAMES = STYLE_NAMES
        self.style_json_file = ""


    def load_pipe(
        self,
        base_model_id: str,
        task_name="txt2img",
        vae_model=None,
        type_model_precision=torch.float16,
        reload=False,
        sdxl_safetensors = False,
        retain_model_in_memory = True,
    ) -> DiffusionPipeline:
        if (
            base_model_id == self.base_model_id
            and task_name == self.task_name
            and hasattr(self, "pipe")
            and self.vae_model == vae_model
            and self.pipe is not None
            and reload == False
        ):
            if self.type_model_precision == type_model_precision or self.device.type == "cpu":
                return

        if hasattr(self, "pipe") and os.path.isfile(base_model_id):
            unload_model = False
            if self.pipe == None:
                unload_model = True
            elif type_model_precision != self.type_model_precision and self.device.type != "cpu":
                unload_model = True
        else:
            if hasattr(self, "pipe"):
                unload_model = False
                if self.pipe == None:
                    unload_model = True
            else:
                unload_model = True
        self.type_model_precision = (
            type_model_precision if torch.cuda.is_available() else torch.float32
        )

        if self.type_model_precision == torch.float32 and os.path.isfile(base_model_id):
            logger.info(f"Working with full precision {str(self.type_model_precision)}")

        # Load model
        if self.base_model_id == base_model_id and self.pipe is not None and reload == False and self.vae_model == vae_model and unload_model == False:
            #logger.info("Previous loaded base model") # not return
            class_name = self.class_name
        else:
            # Unload previous model and stuffs
            self.pipe = None
            self.model_memory = {}
            self.lora_memory = [None, None, None, None, None]
            self.lora_scale_memory = [1.0, 1.0, 1.0, 1.0, 1.0]
            self.LCMconfig = None
            self.embed_loaded = []
            self.FreeU = False
            torch.cuda.empty_cache()
            gc.collect()

            # Load new model
            if os.path.isfile(base_model_id): # exists or not same # if os.path.exists(base_model_id):

                if sdxl_safetensors:
                    logger.info("Default VAE: madebyollin/sdxl-vae-fp16-fix")
                    self.pipe = StableDiffusionXLPipeline.from_single_file(
                        base_model_id,
                        vae=AutoencoderKL.from_pretrained(
                            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
                        ),
                        torch_dtype=self.type_model_precision,
                    )
                    class_name = "StableDiffusionXLPipeline"
                else:
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
            if vae_model is None :
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
                        subfolder = "vae",
                    )
                try:
                  self.pipe.vae.to(self.type_model_precision)
                except:
                  logger.warning(f"VAE: not in {self.type_model_precision}")
            self.vae_model = vae_model

            # Define base scheduler
            self.default_scheduler = copy.deepcopy(self.pipe.scheduler)
            logger.debug(f"Base sampler: {self.default_scheduler}")

        if task_name in self.model_memory:
            self.pipe = self.model_memory[task_name]
            # Create new base values
            #self.pipe.to(self.device)
            # torch.cuda.empty_cache()
            # gc.collect()
            self.base_model_id = base_model_id
            self.task_name = task_name
            self.vae_model = vae_model
            self.class_name = class_name
            self.pipe.watermark = None
            return

        # Load task
        model_id = CONTROLNET_MODEL_IDS[task_name]

        if task_name == "inpaint":
            match class_name:
                case "StableDiffusionPipeline":

                    controlnet = ControlNetModel.from_pretrained(
                        model_id, torch_dtype=self.type_model_precision
                    )

                    self.pipe = StableDiffusionControlNetInpaintPipeline(
                        vae=self.pipe.vae,
                        text_encoder=self.pipe.text_encoder,
                        tokenizer=self.pipe.tokenizer,
                        unet=self.pipe.unet,
                        controlnet=controlnet,
                        scheduler=self.pipe.scheduler,
                        safety_checker=self.pipe.safety_checker,
                        feature_extractor=self.pipe.feature_extractor,
                        requires_safety_checker=self.pipe.config.requires_safety_checker,
                    )
                case "StableDiffusionXLPipeline":

                    self.pipe = StableDiffusionXLInpaintPipeline(
                        vae=self.pipe.vae,
                        text_encoder=self.pipe.text_encoder,
                        text_encoder_2=self.pipe.text_encoder_2,
                        tokenizer=self.pipe.tokenizer,
                        tokenizer_2=self.pipe.tokenizer_2,
                        unet=self.pipe.unet,
                        # controlnet=self.controlnet,
                        scheduler=self.pipe.scheduler,
                    )


        if task_name not in ["txt2img", "inpaint", "img2img"]:
            match class_name:
                case "StableDiffusionPipeline":

                    controlnet = ControlNetModel.from_pretrained(
                        model_id, torch_dtype=self.type_model_precision
                    )

                    self.pipe = StableDiffusionControlNetPipeline(
                        vae=self.pipe.vae,
                        text_encoder=self.pipe.text_encoder,
                        tokenizer=self.pipe.tokenizer,
                        unet=self.pipe.unet,
                        controlnet=controlnet,
                        scheduler=self.pipe.scheduler,
                        safety_checker=self.pipe.safety_checker,
                        feature_extractor=self.pipe.feature_extractor,
                        requires_safety_checker=self.pipe.config.requires_safety_checker,
                    )
                    self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

                case "StableDiffusionXLPipeline":

                    adapter = T2IAdapter.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16,
                        varient="fp16",
                    ).to(self.device)

                    self.pipe = StableDiffusionXLAdapterPipeline(
                        vae=self.pipe.vae,
                        text_encoder=self.pipe.text_encoder,
                        text_encoder_2=self.pipe.text_encoder_2,
                        tokenizer=self.pipe.tokenizer,
                        tokenizer_2=self.pipe.tokenizer_2,
                        unet=self.pipe.unet,
                        adapter=adapter,
                        scheduler=self.pipe.scheduler,
                    ).to(self.device)


        if task_name in ["txt2img", "img2img"]:
            match class_name:

                case "StableDiffusionPipeline":
                    self.pipe = StableDiffusionPipeline(
                        vae=self.pipe.vae,
                        text_encoder=self.pipe.text_encoder,
                        tokenizer=self.pipe.tokenizer,
                        unet=self.pipe.unet,
                        scheduler=self.pipe.scheduler,
                        safety_checker=self.pipe.safety_checker,
                        feature_extractor=self.pipe.feature_extractor,
                        requires_safety_checker=self.pipe.config.requires_safety_checker,
                    )

                case "StableDiffusionXLPipeline":
                    self.pipe = StableDiffusionXLPipeline(
                        vae=self.pipe.vae,
                        text_encoder=self.pipe.text_encoder,
                        text_encoder_2=self.pipe.text_encoder_2,
                        tokenizer=self.pipe.tokenizer,
                        tokenizer_2=self.pipe.tokenizer_2,
                        unet=self.pipe.unet,
                        scheduler=self.pipe.scheduler,
                    )

            if task_name == "img2img":
                self.pipe = AutoPipelineForImage2Image.from_pipe(self.pipe)

        # Create new base values
        self.pipe.to(self.device)
        torch.cuda.empty_cache()
        gc.collect()

        self.base_model_id = base_model_id
        self.task_name = task_name
        self.vae_model = vae_model
        self.class_name = class_name

        if self.class_name == "StableDiffusionXLPipeline":
            self.pipe.enable_vae_slicing()
            self.pipe.enable_vae_tiling()
            self.pipe.watermark = None

        if retain_model_in_memory == True and task_name not in self.model_memory:
            self.model_memory[task_name] = self.pipe

        return

    def load_controlnet_weight(self, task_name: str) -> None:
        torch.cuda.empty_cache()
        gc.collect()
        model_id = CONTROLNET_MODEL_IDS[task_name]
        controlnet = ControlNetModel.from_pretrained(
            model_id, torch_dtype=self.type_model_precision
        )
        controlnet.to(self.device)
        torch.cuda.empty_cache()
        gc.collect()
        self.pipe.controlnet = controlnet
        #self.task_name = task_name

    @torch.autocast("cuda")
    def run_pipe(
        self,
        prompt: str,
        negative_prompt: str,
        prompt_embeds,
        negative_prompt_embeds,
        control_image: PIL.Image.Image,
        num_images: int,
        num_steps: int,
        guidance_scale: float,
        clip_skip: int,
        generator,
        controlnet_conditioning_scale,
        control_guidance_start,
        control_guidance_end,
    ) -> list[PIL.Image.Image]:
        # Return PIL images
        # generator = torch.Generator().manual_seed(seed)
        return self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            clip_skip=clip_skip,
            num_images_per_prompt=num_images,
            num_inference_steps=num_steps,
            generator=generator,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            image=control_image,
        ).images

    @torch.autocast("cuda")
    def run_pipe_SD(
        self,
        prompt: str,
        negative_prompt: str,
        prompt_embeds,
        negative_prompt_embeds,
        num_images: int,
        num_steps: int,
        guidance_scale: float,
        clip_skip: int,
        height: int,
        width: int,
        generator,
    ) -> list[PIL.Image.Image]:
        # Return PIL images
        # generator = torch.Generator().manual_seed(seed)
        self.preview_handle = None
        return self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            clip_skip=clip_skip,
            num_images_per_prompt=num_images,
            num_inference_steps=num_steps,
            generator=generator,
            height=height,
            width=width,
            callback=self.callback_pipe if self.image_previews else None,
            callback_steps=10 if self.image_previews else 100,
        ).images

    # @torch.autocast('cuda')
    # def run_pipe_SDXL(
    #     self,
    #     prompt: str,
    #     negative_prompt: str,
    #     prompt_embeds,
    #     negative_prompt_embeds,
    #     num_images: int,
    #     num_steps: int,
    #     guidance_scale: float,
    #     clip_skip: int,
    #     height : int,
    #     width : int,
    #     generator,
    #     seddd,
    #     conditioning,
    #     pooled,
    # ) -> list[PIL.Image.Image]:
    #     # Return PIL images
    #     #generator = torch.Generator("cuda").manual_seed(seddd) # generator = torch.Generator("cuda").manual_seed(seed),
    #     return self.pipe(
    #         prompt = None,
    #         negative_prompt = None,
    #         prompt_embeds=conditioning[0:1],
    #         pooled_prompt_embeds=pooled[0:1],
    #         negative_prompt_embeds=conditioning[1:2],
    #         negative_pooled_prompt_embeds=pooled[1:2],
    #         height = height,
    #         width = width,
    #         num_inference_steps = num_steps,
    #         guidance_scale = guidance_scale,
    #         clip_skip = clip_skip,
    #         num_images_per_prompt = num_images,
    #         generator = generator,
    #         ).images

    @torch.autocast("cuda")
    def run_pipe_inpaint(
        self,
        prompt: str,
        negative_prompt: str,
        prompt_embeds,
        negative_prompt_embeds,
        control_image: PIL.Image.Image,
        num_images: int,
        num_steps: int,
        guidance_scale: float,
        clip_skip: int,
        strength: float,
        init_image,
        control_mask,
        controlnet_conditioning_scale,
        control_guidance_start,
        control_guidance_end,
        generator,
    ) -> list[PIL.Image.Image]:
        # Return PIL images
        # generator = torch.Generator().manual_seed(seed)
        return self.pipe(
            prompt=None,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            eta=1.0,
            strength=strength,
            image=init_image,  # original image
            mask_image=control_mask,  # mask, values of 0 to 255
            control_image=control_image,  # tensor control image
            num_images_per_prompt=num_images,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            clip_skip=clip_skip,
            generator=generator,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
        ).images

    @torch.autocast("cuda")
    def run_pipe_img2img(
        self,
        prompt: str,
        negative_prompt: str,
        prompt_embeds,
        negative_prompt_embeds,
        num_images: int,
        num_steps: int,
        guidance_scale: float,
        clip_skip: int,
        strength: float,
        init_image,
        generator,
    ) -> list[PIL.Image.Image]:
        # Return PIL images
        # generator = torch.Generator().manual_seed(seed)
        return self.pipe(
            prompt=None,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            eta=1.0,
            strength=strength,
            image=init_image,  # original image
            num_images_per_prompt=num_images,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            clip_skip=clip_skip,
            generator=generator,
        ).images

    ### self.x_process return image_preprocessor###
    @torch.inference_mode()
    def process_canny(
        self,
        image: np.ndarray,
        image_resolution: int,
        preprocess_resolution: int,
        low_threshold: int,
        high_threshold: int,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError

        self.preprocessor.load("Canny")
        control_image = self.preprocessor(
            image=image,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            image_resolution=image_resolution,
            detect_resolution=preprocess_resolution,
        )

        return control_image

    @torch.inference_mode()
    def process_mlsd(
        self,
        image: np.ndarray,
        image_resolution: int,
        preprocess_resolution: int,
        value_threshold: float,
        distance_threshold: float,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError

        self.preprocessor.load("MLSD")
        control_image = self.preprocessor(
            image=image,
            image_resolution=image_resolution,
            detect_resolution=preprocess_resolution,
            thr_v=value_threshold,
            thr_d=distance_threshold,
        )

        return control_image

    @torch.inference_mode()
    def process_scribble(
        self,
        image: np.ndarray,
        image_resolution: int,
        preprocess_resolution: int,
        preprocessor_name: str,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError

        if preprocessor_name == "None":
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        elif preprocessor_name == "HED":
            self.preprocessor.load(preprocessor_name)
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
                scribble=False,
            )
        elif preprocessor_name == "PidiNet":
            self.preprocessor.load(preprocessor_name)
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
                safe=False,
            )

        return control_image

    @torch.inference_mode()
    def process_scribble_interactive(
        self,
        image_and_mask: dict[str, np.ndarray],
        image_resolution: int,
    ) -> list[PIL.Image.Image]:
        if image_and_mask is None:
            raise ValueError

        image = image_and_mask["mask"]
        image = HWC3(image)
        image = resize_image(image, resolution=image_resolution)
        control_image = PIL.Image.fromarray(image)

        return control_image

    @torch.inference_mode()
    def process_softedge(
        self,
        image: np.ndarray,
        image_resolution: int,
        preprocess_resolution: int,
        preprocessor_name: str,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError

        if preprocessor_name == "None":
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        elif preprocessor_name in ["HED", "HED safe"]:
            safe = "safe" in preprocessor_name
            self.preprocessor.load("HED")
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
                scribble=safe,
            )
        elif preprocessor_name in ["PidiNet", "PidiNet safe"]:
            safe = "safe" in preprocessor_name
            self.preprocessor.load("PidiNet")
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
                safe=safe,
            )
        else:
            raise ValueError

        return control_image

    @torch.inference_mode()
    def process_openpose(
        self,
        image: np.ndarray,
        image_resolution: int,
        preprocess_resolution: int,
        preprocessor_name: str,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError

        if preprocessor_name == "None":
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        else:
            self.preprocessor.load("Openpose")
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
                hand_and_face=True,
            )

        return control_image

    @torch.inference_mode()
    def process_segmentation(
        self,
        image: np.ndarray,
        image_resolution: int,
        preprocess_resolution: int,
        preprocessor_name: str,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError

        if preprocessor_name == "None":
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        else:
            self.preprocessor.load(preprocessor_name)
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
            )

        return control_image

    @torch.inference_mode()
    def process_depth(
        self,
        image: np.ndarray,
        image_resolution: int,
        preprocess_resolution: int,
        preprocessor_name: str,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError

        if preprocessor_name == "None":
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        else:
            self.preprocessor.load(preprocessor_name)
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
            )

        return control_image

    @torch.inference_mode()
    def process_normal(
        self,
        image: np.ndarray,
        image_resolution: int,
        preprocess_resolution: int,
        preprocessor_name: str,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError

        if preprocessor_name == "None":
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        else:
            self.preprocessor.load("NormalBae")
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
            )

        return control_image

    @torch.inference_mode()
    def process_lineart(
        self,
        image: np.ndarray,
        image_resolution: int,
        preprocess_resolution: int,
        preprocessor_name: str,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError

        if preprocessor_name in ["None", "None (anime)"]:
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        elif preprocessor_name in ["Lineart", "Lineart coarse"]:
            coarse = "coarse" in preprocessor_name
            self.preprocessor.load("Lineart")
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
                coarse=coarse,
            )
        elif preprocessor_name == "Lineart (anime)":
            self.preprocessor.load("LineartAnime")
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
            )

        if self.class_name == "StableDiffusionPipeline":
            if "anime" in preprocessor_name:
                self.load_controlnet_weight("lineart_anime")
                logger.info("Linear anime")
            else:
                self.load_controlnet_weight("lineart")

        return control_image

    @torch.inference_mode()
    def process_shuffle(
        self,
        image: np.ndarray,
        image_resolution: int,
        preprocessor_name: str,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError

        if preprocessor_name == "None":
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        else:
            self.preprocessor.load(preprocessor_name)
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
            )

        return control_image

    @torch.inference_mode()
    def process_ip2p(
        self,
        image: np.ndarray,
        image_resolution: int,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError

        image = HWC3(image)
        image = resize_image(image, resolution=image_resolution)
        control_image = PIL.Image.fromarray(image)

        return control_image

    @torch.inference_mode()
    def process_inpaint(
        self,
        image: np.ndarray,
        image_resolution: int,
        preprocess_resolution: int,
        image_mask: str,  ###
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError

        image = HWC3(image)
        image = resize_image(image, resolution=image_resolution)
        init_image = PIL.Image.fromarray(image)

        image_mask = HWC3(image_mask)
        image_mask = resize_image(image_mask, resolution=image_resolution)
        control_mask = PIL.Image.fromarray(image_mask)

        control_image = make_inpaint_condition(init_image, control_mask)

        return init_image, control_mask, control_image

    @torch.inference_mode()
    def process_img2img(
        self,
        image: np.ndarray,
        image_resolution: int,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError

        image = HWC3(image)
        image = resize_image(image, resolution=image_resolution)
        init_image = PIL.Image.fromarray(image)

        return init_image

    def get_scheduler(self, name):
        if name in SCHEDULER_CONFIG_MAP:
            scheduler_class, config = SCHEDULER_CONFIG_MAP[name]
            #return scheduler_class.from_config(self.pipe.scheduler.config, **config)
            # beta self.default_scheduler
            return scheduler_class.from_config(self.default_scheduler.config, **config)
        else:
            raise ValueError(f"Scheduler with name {name} not found. Valid schedulers: {', '.join(scheduler_names)}")

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
            except:
                prompt_ti = prompt
                negative_prompt_ti = negative_prompt
                logger.error("FAILED: Convert prompt for textual inversion")

            # prompt syntax style a1...
            if syntax_weights == "Classic":
                self.pipe.to("cuda")
                prompt_ti = get_embed_new(prompt_ti, self.pipe, self.compel, only_convert_string=True)
                negative_prompt_ti = get_embed_new(negative_prompt_ti, self.pipe, self.compel, only_convert_string=True)
            else:
                prompt_ti = prompt
                negative_prompt_ti = negative_prompt

            conditioning, pooled = self.compel([prompt_ti, negative_prompt_ti])

            return conditioning, pooled



    def process_lora(self, select_lora, lora_weights_scale, unload=False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not unload:
            if select_lora != None:
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
                    logger.error(f"ERROR: LoRA not compatible: {select_lora}")
                    logger.debug(f"{str(e)}")
            return self.pipe
        else:
            # Unload numerically unstable but fast and need less memory
            if select_lora != None:
                try:
                    self.pipe = lora_mix_load(
                        self.pipe,
                        select_lora,
                        -lora_weights_scale,
                        device=device,
                        dtype=self.type_model_precision,
                    )
                    logger.debug(f"Unload LoRA: {select_lora}")
                except:
                    pass
            return self.pipe

    def load_style_file(self, style_json_file):
        if os.path.exists(style_json_file):
            try:
                file_json_read = get_json_content(style_json_file)
                self.styles_data = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in file_json_read}
                self.STYLE_NAMES = list(self.styles_data.keys())
                self.style_json_file = style_json_file
                logger.info(f"Styles json file loaded with {len(self.STYLE_NAMES)} styles")
                logger.debug(str(self.STYLE_NAMES))
            except Exception as e:
                logger.error(str(e))
        else:
            logger.error("Not found styles json file in directory")

    def callback_pipe(self, iter, t, latents):
        # convert latents to image
        with torch.no_grad():
            latents = 1 / 0.18215 * latents
            image = self.pipe.vae.decode(latents).sample

            image = (image / 2 + 0.5).clamp(0, 1)

            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()

            # convert to PIL Images
            image = self.pipe.numpy_to_pil(image)

            # show one image
            # global preview_handle
            if self.preview_handle == None:
                self.preview_handle = display(image[0], display_id=True)
            else:
                self.preview_handle.update(image[0])

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
        controlnet_conditioning_scale: float = 1.0,
        control_guidance_start: float = 0.0,
        control_guidance_end: float = 1.0,
        t2i_adapter_preprocessor: bool = True,
        t2i_adapter_conditioning_scale: float = 1.0,
        t2i_adapter_conditioning_factor: float = 1.0,

        upscaler_model_path: Optional[str] = None, # add latent
        upscaler_increases_size: float = 1.5,
        esrgan_tile: int = 100,
        esrgan_tile_overlap: int = 10,
        hires_steps: int = 25,
        hires_denoising_strength: float = 0.35,
        hires_prompt: str = "",
        hires_negative_prompt: str = "",
        hires_sampler: str = "Use same sampler",

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
                The sampler used for the generation process. Available samplers: DPM++ 2M, DPM++ 2M Karras, DPM++ 2M SDE,
                DPM++ 2M SDE Karras, DPM++ SDE, DPM++ SDE Karras, DPM2, DPM2 Karras, Euler, Euler a, Heun, LMS, LMS Karras,
                DDIM, DEIS, UniPC, DPM2 a, DPM2 a Karras, PNDM, LCM, DPM++ 2M Lu, DPM++ 2M Ef, DPM++ 2M SDE Lu and DPM++ 2M SDE Ef.
            syntax_weights (str, optional, defaults to "Classic"):
                Specifies the type of syntax weights used during generation. "Classic" is (word:weight), "Compel" is (word)weight
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
            upscaler_model_path (str, optional):
                Placeholder for upscaler model path.
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

            Additional parameters that will be used in ControlNet for SD 1.5 depending on the task:
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

        if self.task_name != "txt2img" and image == None:
            raise ValueError(
                "You need to specify the <image> for this task."
            )
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
                "Control guidance start (ControlNet Start Threshold) cannot be larger or equal to control guidance end (ControlNet Stop Threshold). The default values 0.0 and 1.0 will be used."
            )
            control_guidance_start, control_guidance_end = 0.0, 1.0

        self.gui_active = gui_active
        self.image_previews = image_previews

        if self.pipe == None:
            self.load_pipe(
                self.base_model_id,
                task_name=self.task_name,
                vae_model=self.vae_model,
                reload=True,
            )

        self.pipe.set_progress_bar_config(leave=leave_progress_bar)
        self.pipe.set_progress_bar_config(disable=disable_progress_bar)

        xformers_memory_efficient_attention=False # disabled
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
            prompt, negative_prompt = apply_style(style_prompt, prompt, negative_prompt, self.styles_data, self.STYLE_NAMES)

        # LoRA load
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
                if single_lora != None:
                    logger.info(f"LoRA in memory: {single_lora}")
            pass

        else:
            logger.debug("_un, re and load_ lora")
            self.pipe = self.process_lora(
                self.lora_memory[0], self.lora_scale_memory[0], unload=True
            )
            self.pipe = self.process_lora(
                self.lora_memory[1], self.lora_scale_memory[1], unload=True
            )
            self.pipe = self.process_lora(
                self.lora_memory[2], self.lora_scale_memory[2], unload=True
            )
            self.pipe = self.process_lora(
                self.lora_memory[3], self.lora_scale_memory[3], unload=True
            )
            self.pipe = self.process_lora(
                self.lora_memory[4], self.lora_scale_memory[4], unload=True
            )

            self.pipe = self.process_lora(lora_A, lora_scale_A)
            self.pipe = self.process_lora(lora_B, lora_scale_B)
            self.pipe = self.process_lora(lora_C, lora_scale_C)
            self.pipe = self.process_lora(lora_D, lora_scale_D)
            self.pipe = self.process_lora(lora_E, lora_scale_E)

        self.lora_memory = [lora_A, lora_B, lora_C, lora_D, lora_E]
        self.lora_scale_memory = [
            lora_scale_A,
            lora_scale_B,
            lora_scale_C,
            lora_scale_D,
            lora_scale_E,
        ]

        # LCM config
        if sampler == "LCM" and self.LCMconfig == None:
            if self.class_name == "StableDiffusionPipeline":
                adapter_id = "latent-consistency/lcm-lora-sdv1-5"
            elif self.class_name == "StableDiffusionXLPipeline":
                adapter_id = "latent-consistency/lcm-lora-sdxl"

            self.process_lora(adapter_id, 1.0)
            self.LCMconfig = adapter_id
            logger.info("LCM")
        elif sampler != "LCM" and self.LCMconfig != None:
            self.process_lora(self.LCMconfig, 1.0, unload=True)
            self.LCMconfig = None
        elif self.LCMconfig != None:
            logger.info("LCM")

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
            #self.pipe.scheduler = DPMSolverSinglestepScheduler() # fix default params by random scheduler, not recomn
            self.pipe.scheduler = self.get_scheduler(sampler)
        except Exception as e:
            logger.debug(f"{e}")
            logger.warning(f"Error in sampler, please try again")
            #self.pipe = None
            torch.cuda.empty_cache()
            gc.collect()
            return

        self.pipe.safety_checker = None

        # Get image Global
        if self.task_name != "txt2img":
            if isinstance(image, str):
                # If the input is a string (file path), open it as an image
                image_pil = Image.open(image)
                numpy_array = np.array(image_pil, dtype=np.uint8)
            elif isinstance(image, Image.Image):
                # If the input is already a PIL Image, convert it to a NumPy array
                numpy_array = np.array(image, dtype=np.uint8)
            elif isinstance(image, np.ndarray):
                # If the input is a NumPy array, np.uint8
                numpy_array = image.astype(np.uint8)
            else:
                if gui_active:
                    logger.info(
                        "Not found image"
                    )
                    return
                else:
                    raise ValueError(
                        "Unsupported image type or not control image found; Bug report to https://github.com/R3gm/stablepy or https://github.com/R3gm/SD_diffusers_interactive"
                    )

            # Extract the RGB channels
            try:
                array_rgb = numpy_array[:, :, :3]
            except:
                logger.error("Unsupported image type")
                raise ValueError(
                    "Unsupported image type; Bug report to https://github.com/R3gm/stablepy or https://github.com/R3gm/SD_diffusers_interactive"
                )  # return

        # Get params preprocess Global SD 1.5
        preprocess_params_config = {}
        if self.task_name not in ["txt2img", "inpaint", "img2img"]:
            preprocess_params_config["image"] = array_rgb
            preprocess_params_config["image_resolution"] = image_resolution

            if self.task_name != "ip2p":
                if self.task_name != "shuffle":
                    preprocess_params_config[
                        "preprocess_resolution"
                    ] = preprocess_resolution
                if self.task_name != "mlsd" and self.task_name != "canny":
                    preprocess_params_config["preprocessor_name"] = preprocessor_name

        # RUN Preprocess SD 1.5
        if self.task_name == "inpaint":
            # Get mask for Inpaint
            if gui_active or os.path.exists(str(image_mask)):
                # Read image mask from gui
                mask_control_img = Image.open(image_mask)
                numpy_array_mask = np.array(mask_control_img, dtype=np.uint8)
                array_rgb_mask = numpy_array_mask[:, :, :3]
            elif not gui_active:
                # Convert control image to draw
                import base64
                import matplotlib.pyplot as plt
                name_without_extension = os.path.splitext(image.split("/")[-1])[0]
                image64 = base64.b64encode(open(image, "rb").read())
                image64 = image64.decode("utf-8")
                img = np.array(plt.imread(f"{image}")[:, :, :3])

                # Create mask interactive
                logger.info(f"Draw the mask on this canvas using the mouse. When you finish, press 'Finish' in the bottom side of the canvas.")
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
                mask_control_img = Image.open(mask_control)
                numpy_array_mask = np.array(mask_control_img, dtype=np.uint8)
                array_rgb_mask = numpy_array_mask[:, :, :3]
            else:
                raise ValueError("No images found")

            init_image, control_mask, control_image = self.process_inpaint(
                image=array_rgb,
                image_resolution=image_resolution,
                preprocess_resolution=preprocess_resolution,  # Not used
                image_mask=array_rgb_mask,
            )

        elif self.task_name == "openpose":
            logger.info("Openpose")
            control_image = self.process_openpose(**preprocess_params_config)

        elif self.task_name == "canny":
            logger.info("Canny")
            control_image = self.process_canny(
                **preprocess_params_config,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
            )

        elif self.task_name == "mlsd":
            logger.info("MLSD")
            control_image = self.process_mlsd(
                **preprocess_params_config,
                value_threshold=value_threshold,
                distance_threshold=distance_threshold,
            )

        elif self.task_name == "scribble":
            logger.info("Scribble")
            control_image = self.process_scribble(**preprocess_params_config)

        elif self.task_name == "softedge":
            logger.info("Softedge")
            control_image = self.process_softedge(**preprocess_params_config)

        elif self.task_name == "segmentation":
            logger.info("Segmentation")
            control_image = self.process_segmentation(**preprocess_params_config)

        elif self.task_name == "depth":
            logger.info("Depth")
            control_image = self.process_depth(**preprocess_params_config)

        elif self.task_name == "normalbae":
            logger.info("NormalBae")
            control_image = self.process_normal(**preprocess_params_config)

        elif self.task_name == "lineart":
            logger.info("Lineart")
            control_image = self.process_lineart(**preprocess_params_config)

        elif self.task_name == "shuffle":
            logger.info("Shuffle")
            control_image = self.process_shuffle(**preprocess_params_config)

        elif self.task_name == "ip2p":
            logger.info("Ip2p")
            control_image = self.process_ip2p(**preprocess_params_config)

        elif self.task_name == "img2img":
            preprocess_params_config["image"] = array_rgb
            preprocess_params_config["image_resolution"] = image_resolution
            init_image = self.process_img2img(**preprocess_params_config)

        # RUN Preprocess T2I for SDXL
        if self.class_name == "StableDiffusionXLPipeline":
            # Get params preprocess XL
            preprocess_params_config_xl = {}
            if self.task_name not in ["txt2img", "inpaint", "img2img"]:
                preprocess_params_config_xl["image"] = array_rgb
                preprocess_params_config_xl["preprocess_resolution"] = preprocess_resolution
                preprocess_params_config_xl["image_resolution"] = image_resolution
                # preprocess_params_config_xl["additional_prompt"] = additional_prompt # ""

            if self.task_name == "sdxl_canny": # preprocessor true default
                logger.info("SDXL Canny: Preprocessor active by default")
                control_image = self.process_canny(
                    **preprocess_params_config_xl,
                    low_threshold=low_threshold,
                    high_threshold=high_threshold,
                )
            elif self.task_name == "sdxl_openpose":
                logger.info("SDXL Openpose")
                control_image = self.process_openpose(
                    preprocessor_name = "Openpose" if t2i_adapter_preprocessor else "None",
                    **preprocess_params_config_xl,
                )
            elif self.task_name == "sdxl_sketch":
                logger.info("SDXL Scribble")
                control_image = self.process_scribble(
                    preprocessor_name = "PidiNet" if t2i_adapter_preprocessor else "None",
                    **preprocess_params_config_xl,
                )
            elif self.task_name == "sdxl_depth-midas":
                logger.info("SDXL Depth")
                control_image = self.process_depth(
                    preprocessor_name = "Midas" if t2i_adapter_preprocessor else "None",
                    **preprocess_params_config_xl,
                )
            elif self.task_name == "sdxl_lineart":
                logger.info("SDXL Lineart")
                control_image = self.process_lineart(
                    preprocessor_name = "Lineart" if t2i_adapter_preprocessor else "None",
                    **preprocess_params_config_xl,
                )

        # Get general params for TASK
        if self.class_name == "StableDiffusionPipeline":
            # Base params pipe sd
            pipe_params_config = {
                "prompt": None,  # prompt, 
                "negative_prompt": None,  # negative_prompt,
                "prompt_embeds": prompt_emb,
                "negative_prompt_embeds": negative_prompt_emb,
                "num_images": num_images,
                "num_steps": num_steps,
                "guidance_scale": guidance_scale,
                "clip_skip": None,  # clip_skip, because we use clip skip of compel
            }
        else:
            # Base params pipe sdxl
            pipe_params_config = {
                "prompt" : None,
                "negative_prompt" : None,
                "num_inference_steps" : num_steps,
                "guidance_scale" : guidance_scale,
                "clip_skip" : None,
                "num_images_per_prompt" : num_images,
            }

        # New params
        if self.class_name == "StableDiffusionXLPipeline":
            # pipe sdxl
            if self.task_name == "txt2img":
                pipe_params_config["height"] = img_height
                pipe_params_config["width"] = img_width
            elif self.task_name == "inpaint":
                pipe_params_config["strength"] = strength
                pipe_params_config["image"] = init_image
                pipe_params_config["mask_image"] = control_mask
                logger.info(f"Image resolution: {str(init_image.size)}")
            elif self.task_name not in ["txt2img", "inpaint", "img2img"]:
                pipe_params_config["image"] = control_image
                pipe_params_config["adapter_conditioning_scale"] = t2i_adapter_conditioning_scale
                pipe_params_config["adapter_conditioning_factor"] = t2i_adapter_conditioning_factor
                logger.info(f"Image resolution: {str(control_image.size)}")
            elif self.task_name == "img2img":
                pipe_params_config["strength"] = strength
                pipe_params_config["image"] = init_image
                logger.info(f"Image resolution: {str(init_image.size)}")
        elif self.task_name == "txt2img":
            pipe_params_config["height"] = img_height
            pipe_params_config["width"] = img_width
        elif self.task_name == "inpaint":
            pipe_params_config["strength"] = strength
            pipe_params_config["init_image"] = init_image
            pipe_params_config["control_mask"] = control_mask
            pipe_params_config["control_image"] = control_image
            pipe_params_config[
                "controlnet_conditioning_scale"
            ] = controlnet_conditioning_scale
            pipe_params_config["control_guidance_start"] = control_guidance_start
            pipe_params_config["control_guidance_end"] = control_guidance_end
            logger.info(f"Image resolution: {str(init_image.size)}")
        elif self.task_name not in ["txt2img", "inpaint", "img2img"]:
            pipe_params_config["control_image"] = control_image
            pipe_params_config[
                "controlnet_conditioning_scale"
            ] = controlnet_conditioning_scale
            pipe_params_config["control_guidance_start"] = control_guidance_start
            pipe_params_config["control_guidance_end"] = control_guidance_end
            logger.info(f"Image resolution: {str(control_image.size)}")
        elif self.task_name == "img2img":
            pipe_params_config["strength"] = strength
            pipe_params_config["init_image"] = init_image
            logger.info(f"Image resolution: {str(init_image.size)}")

        # detailfix params and pipe global
        if adetailer_A or adetailer_B:

            # global params detailfix
            default_params_detailfix = {
                "face_detector_ad" : True,
                "person_detector_ad" : True,
                "hand_detector_ad" : False,
                "prompt": "",
                "negative_prompt" : "",
                "strength" : 0.35,
                "mask_dilation" : 4,
                "mask_blur" : 4,
                "mask_padding" : 32,
                #"sampler" : "Use same sampler",
                #"inpaint_only" : True,
            }

            # Pipe detailfix_pipe
            if not hasattr(self, "detailfix_pipe") or not retain_detailfix_model_previous_load:
                if  adetailer_A_params.get("inpaint_only", False) == True or adetailer_B_params.get("inpaint_only", False) == True:
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
            if  adetailer_A_params.get("sampler", "Use same sampler") != "Use same sampler":
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

        if adetailer_A:
            for key_param, default_value in default_params_detailfix.items():
                if key_param not in adetailer_A_params:
                    adetailer_A_params[key_param] = default_value
                elif type(default_value) != type(adetailer_A_params[key_param]):
                    logger.warning(f"DetailFix A: Error type param, set default {str(key_param)}")
                    adetailer_A_params[key_param] = default_value

            detailfix_params_A = {
                "prompt": adetailer_A_params["prompt"],
                "negative_prompt" : adetailer_A_params["negative_prompt"],
                "strength" : adetailer_A_params["strength"],
                "num_inference_steps" : num_steps,
                "guidance_scale" : guidance_scale,
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

        if adetailer_B:
            for key_param, default_value in default_params_detailfix.items():
                if key_param not in adetailer_B_params:
                    adetailer_B_params[key_param] = default_value
                elif type(default_value) != type(adetailer_B_params[key_param]):
                    logger.warning(f"DetailfFix B: Error type param, set default {str(key_param)}")
                    adetailer_B_params[key_param] = default_value

            detailfix_params_B = {
                "prompt": adetailer_B_params["prompt"],
                "negative_prompt" : adetailer_B_params["negative_prompt"],
                "strength" : adetailer_B_params["strength"],
                "num_inference_steps" : num_steps,
                "guidance_scale" : guidance_scale,
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

        if hires_steps > 1 and upscaler_model_path != None:
            # Hires params BASE
            hires_params_config = {
                "prompt" : None,
                "negative_prompt" : None,
                "num_inference_steps" : hires_steps,
                "guidance_scale" : guidance_scale,
                "clip_skip" : None,
                "strength" : hires_denoising_strength,
            }
            if self.class_name == "StableDiffusionPipeline":
                hires_params_config["eta"] = 1.0

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
            if  hires_sampler != "Use same sampler":
                logger.debug("New hires sampler")
                hires_pipe.scheduler = self.get_scheduler(hires_sampler)

            hires_pipe.set_progress_bar_config(leave=leave_progress_bar)
            hires_pipe.set_progress_bar_config(disable=disable_progress_bar)
            hires_pipe.to(self.device)
            torch.cuda.empty_cache()
            gc.collect()
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
            if hires_steps > 1 and upscaler_model_path != None:
                logger.debug(f"scheduler_hires: {hires_pipe.scheduler}")
        except Exception as e:
            logger.debug(f"{str(e)}")

        # === RUN PIPE === #
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
                    except:
                        logger.warning("Generator in CPU")
                        generator = torch.Generator().manual_seed(calculate_seed)

                generators.append(generator)

            # fix img2img bug need concat tensor prompts with generator same number (only in batch inference)
            pipe_params_config["generator"] = generators if self.task_name != "img2img" else generators[0] # no list
            seeds = seeds if self.task_name != "img2img" else [seeds[0]] * num_images

            try:
                if self.class_name == "StableDiffusionXLPipeline":
                    # sdxl pipe
                    images = self.pipe(
                        prompt_embeds=conditioning[0:1],
                        pooled_prompt_embeds=pooled[0:1],
                        negative_prompt_embeds=conditioning[1:2],
                        negative_pooled_prompt_embeds=pooled[1:2],
                        #generator=pipe_params_config["generator"],
                        **pipe_params_config,
                    ).images
                    if self.task_name not in ["txt2img", "inpaint", "img2img"]:
                        images = [control_image] + images
                elif self.task_name == "txt2img":
                    images = self.run_pipe_SD(**pipe_params_config)
                elif self.task_name == "inpaint":
                    images = self.run_pipe_inpaint(**pipe_params_config)
                elif self.task_name not in ["txt2img", "inpaint", "img2img"]:
                    results = self.run_pipe(
                        **pipe_params_config
                    )  ## pipe ControlNet add condition_weights
                    images = [control_image] + results
                    del results
                elif self.task_name == "img2img":
                    images = self.run_pipe_img2img(**pipe_params_config)
            except Exception as e:
                e = str(e)
                if "Tensor with 2 elements cannot be converted to Scalar" in e:
                    logger.debug(e)
                    logger.error("Error in sampler; trying with DDIM sampler")
                    self.pipe.scheduler = self.default_scheduler
                    self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
                    if self.class_name == "StableDiffusionXLPipeline":
                        # sdxl pipe
                        images = self.pipe(
                            prompt_embeds=conditioning[0:1],
                            pooled_prompt_embeds=pooled[0:1],
                            negative_prompt_embeds=conditioning[1:2],
                            negative_pooled_prompt_embeds=pooled[1:2],
                            #generator=pipe_params_config["generator"],
                            **pipe_params_config,
                        ).images
                        if self.task_name not in ["txt2img", "inpaint", "img2img"]:
                            images = [control_image] + images
                    elif self.task_name == "txt2img":
                        images = self.run_pipe_SD(**pipe_params_config)
                    elif self.task_name == "inpaint":
                        images = self.run_pipe_inpaint(**pipe_params_config)
                    elif self.task_name not in ["txt2img", "inpaint", "img2img"]:
                        results = self.run_pipe(
                            **pipe_params_config
                        )  ## pipe ControlNet add condition_weights
                        images = [control_image] + results
                        del results
                    elif self.task_name == "img2img":
                        images = self.run_pipe_img2img(**pipe_params_config)
                elif "The size of tensor a (0) must match the size of tensor b (3) at non-singleton" in e:
                    raise ValueError(f"steps / strength too low for the model to produce a satisfactory response")
                else:
                    raise ValueError(e)
                    
            torch.cuda.empty_cache()
            gc.collect()

            if hires_before_adetailer and upscaler_model_path != None:
                logger.debug(f"Hires before; same seed for each image (no batch)")
                images = process_images_high_resolution(
                    images,
                    upscaler_model_path,
                    upscaler_increases_size,
                    esrgan_tile, esrgan_tile_overlap,
                    hires_steps, hires_params_config,
                    self.task_name,
                    generators[0], #pipe_params_config["generator"][0], # no generator
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
                        pipe_params_df=detailfix_params_A,
                        detailfix_pipe=detailfix_pipe,
                        image_list_task=images,
                        **adetailer_A_params,
                    )
                if adetailer_B:
                    images = ad_model_process(
                        pipe_params_df=detailfix_params_B,
                        detailfix_pipe=detailfix_pipe,
                        image_list_task=images,
                        **adetailer_B_params,
                    )

                if self.task_name not in ["txt2img", "inpaint", "img2img"]:
                    images = [control_image] + images
                # del detailfix_pipe
                torch.cuda.empty_cache()
                gc.collect()

            if hires_after_adetailer and upscaler_model_path != None:
                logger.debug(f"Hires after; same seed for each image (no batch)")
                images = process_images_high_resolution(
                    images,
                    upscaler_model_path,
                    upscaler_increases_size,
                    esrgan_tile, esrgan_tile_overlap,
                    hires_steps, hires_params_config,
                    self.task_name,
                    generators[0], #pipe_params_config["generator"][0], # no generator
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
            metadata = [
                prompt,
                negative_prompt,
                self.base_model_id,
                self.vae_model,
                num_steps,
                guidance_scale,
                sampler,
                0000000000, #calculate_seed,
                img_width,
                img_height,
                clip_skip,
            ]

            valid_seeds = [0] + seeds if self.task_name not in ["txt2img", "inpaint", "img2img"] else seeds
            for image_, seed_ in zip(images, valid_seeds):
                image_path = "not saved in storage"
                if save_generated_images:
                    metadata[7] = seed_
                    image_path = save_pil_image_with_metadata(image_, image_storage_location, metadata)
                image_list.append(image_path)

            torch.cuda.empty_cache()
            gc.collect()

            if image_list[0] != "not saved in storage":
                logger.info(image_list)

        if hasattr(self, "compel") and not retain_compel_previous_load:
          del self.compel
        torch.cuda.empty_cache()
        gc.collect()

        return images, image_list
