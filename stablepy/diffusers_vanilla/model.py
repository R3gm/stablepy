import gc
import time
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
)
import json
from huggingface_hub import hf_hub_download
import torch
import random
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
)
from .prompt_weights import get_embed_new, add_comma_after_pattern_ti
from .utils import save_pil_image_with_metadata
from .lora_loader import lora_mix_load
from .inpainting_canvas import draw, make_inpaint_condition
from .adetailer import ad_model_process
from ..upscalers.esrgan import UpscalerESRGAN
import os
from compel import Compel
from compel import ReturnedEmbeddingsType
import ipywidgets as widgets, mediapy
from IPython.display import display
from PIL import Image
from asdff.sd import AdCnPreloadPipe
from typing import Union, Optional, List, Tuple, Dict, Any, Callable

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
}


# def download_all_controlnet_weights() -> None:
#     for model_id in CONTROLNET_MODEL_IDS.values():
#         ControlNetModel.from_pretrained(model_id)


class Model_Diffusers:
    def __init__(
        self,
        base_model_id: str = "runwayml/stable-diffusion-v1-5",
        task_name: str = "txt2img",
        vae_model=None,
        type_model_precision=torch.float16,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.base_model_id = ""
        self.task_name = ""
        self.vae_model = None
        self.type_model_precision = (
            type_model_precision if torch.cuda.is_available() else torch.float32
        )  # For SD 1.5

        self.load_pipe(
            base_model_id, task_name, vae_model, type_model_precision
        )
        self.preprocessor = Preprocessor()



    def load_pipe(
        self,
        base_model_id: str,
        task_name="txt2img",
        vae_model=None,
        type_model_precision=torch.float16,
        reload=False,
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
            print("Working with full precision")
        
        # Load model
        if self.base_model_id == base_model_id and self.pipe is not None and reload == False and self.vae_model == vae_model and unload_model == False:
            print("Previous loaded base model") # not return
            class_name = self.class_name
        else:
            # Unload previous model and stuffs
            self.pipe = None
            self.lora_memory = [None, None, None, None, None]
            self.lora_scale_memory = [1.0, 1.0, 1.0, 1.0, 1.0]
            self.embed_loaded = []
            self.FreeU = False
            torch.cuda.empty_cache()
            gc.collect()

            # Load new model
            if os.path.isfile(base_model_id): # exists or not same # if os.path.exists(base_model_id):

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
                        print("Default VAE: madebyollin/sdxl-vae-fp16-fix")
                        self.pipe = DiffusionPipeline.from_pretrained(
                            base_model_id,
                            vae=AutoencoderKL.from_pretrained(
                                "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
                            ),
                            torch_dtype=torch.float16,
                            use_safetensors=True,
                            variant="fp16",
                        )

            # Load VAE after loaded model
            if vae_model is None :
                #print("Default vae")
                pass
            elif class_name == "StableDiffusionPipeline":
                if os.path.isfile(vae_model):
                    self.pipe.vae = AutoencoderKL.from_single_file(
                        vae_model
                    )
                else:
                    self.pipe.vae = AutoencoderKL.from_pretrained(
                        vae_model,
                        subfolder = "vae",
                    )

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


        if task_name != "txt2img" and task_name != "inpaint":
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


        if task_name == "txt2img":
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


        # Create new base values
        self.pipe.to(self.device)
        torch.cuda.empty_cache()
        gc.collect()

        self.base_model_id = base_model_id
        self.task_name = task_name
        self.vae_model = vae_model
        self.class_name = class_name
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

    def get_prompt(self, prompt: str, additional_prompt: str) -> str:
        if not prompt:
            prompt = additional_prompt
        else:
            prompt = f"{prompt}, {additional_prompt}"
        return prompt

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
                print("Linear anime")
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

    def get_scheduler(self, name):
        # Get scheduler
        match name:
            case "DPM++ 2M":
                return DPMSolverMultistepScheduler.from_config(
                    self.pipe.scheduler.config
                )

            case "DPM++ 2M Karras":
                return DPMSolverMultistepScheduler.from_config(
                    self.pipe.scheduler.config, use_karras_sigmas=True
                )

            case "DPM++ 2M SDE":
                return DPMSolverMultistepScheduler.from_config(
                    self.pipe.scheduler.config, algorithm_type="sde-dpmsolver++"
                )

            case "DPM++ 2M SDE Karras":
                return DPMSolverMultistepScheduler.from_config(
                    self.pipe.scheduler.config,
                    use_karras_sigmas=True,
                    algorithm_type="sde-dpmsolver++",
                )

            case "DPM++ SDE":
                return DPMSolverSinglestepScheduler.from_config(
                    self.pipe.scheduler.config,
                )

            case "DPM++ SDE Karras":
                return DPMSolverSinglestepScheduler.from_config(
                    self.pipe.scheduler.config, use_karras_sigmas=True
                )

            case "DPM2":
                return KDPM2DiscreteScheduler.from_config(
                    self.pipe.scheduler.config,
                )

            case "DPM2 Karras":
                return KDPM2DiscreteScheduler.from_config(
                    self.pipe.scheduler.config, use_karras_sigmas=True
                )

            case "Euler":
                return EulerDiscreteScheduler.from_config(
                    self.pipe.scheduler.config,
                )

            case "Euler a":
                return EulerAncestralDiscreteScheduler.from_config(
                    self.pipe.scheduler.config,
                )

            case "Heun":
                return HeunDiscreteScheduler.from_config(
                    self.pipe.scheduler.config,
                )

            case "LMS":
                return LMSDiscreteScheduler.from_config(
                    self.pipe.scheduler.config,
                )

            case "LMS Karras":
                return LMSDiscreteScheduler.from_config(
                    self.pipe.scheduler.config, use_karras_sigmas=True
                )

            case "DDIM":
                return DDIMScheduler.from_config(self.pipe.scheduler.config)

            case "DEISMultistep":
                return DEISMultistepScheduler.from_config(self.pipe.scheduler.config)

            case "UniPCMultistep":
                return UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

    def create_prompt_embeds(
        self,
        prompt,
        negative_prompt,
        textual_inversion,
        clip_skip,
        syntax_weights,
        ):

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
                        print(f"Applied : {name}")

                except ValueError:
                    #print(f"Previous loaded embed {name}")
                    pass
                except Exception as e:
                    print(str(e))
                    print(f"Can't apply embed {name}")
            self.embed_loaded = textual_inversion

        # Clip skip
        # clip_skip_diffusers = None #clip_skip - 1 # future update
        compel = Compel(
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
            prompt_emb = get_embed_new(prompt_ti, self.pipe, compel)
            negative_prompt_emb = get_embed_new(negative_prompt_ti, self.pipe, compel)
        else:
            prompt_emb = get_embed_new(prompt_ti, self.pipe, compel, compel_process_sd=True)
            negative_prompt_emb = get_embed_new(negative_prompt_ti, self.pipe, compel, compel_process_sd=True)

        # Fix error shape
        if prompt_emb.shape != negative_prompt_emb.shape:
            (
                prompt_emb,
                negative_prompt_emb,
            ) = compel.pad_conditioning_tensors_to_same_length(
                [prompt_emb, negative_prompt_emb]
            )

        return prompt_emb, negative_prompt_emb

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
                    print(select_lora)
                except:
                    print(f"ERROR: LoRA not compatible: {select_lora}")
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
                    # print(select_lora, 'unload')
                except:
                    pass
            return self.pipe

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
        adetailer_active: bool = False,
        adetailer_params: Dict[str, Any] = {},
        additional_prompt: str = "",
        upscaler_model_path: Optional[str] = None,
        upscaler_increases_size: float = 1.5,

        image: Optional[Any] = None,
        preprocessor_name: Optional[str] = "None",
        preprocess_resolution: int = 512,
        image_resolution: int = 512,
        image_mask: Optional[Any] = None,
        strength: float = 1.0,
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

        loop_generation: int = 1,
        generator_in_cpu: bool = False,
        leave_progress_bar: bool = False,
        disable_progress_bar: bool = False,
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
                DDIM, DEISMultistep and UniPCMultistep
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
            adetailer_active (bool, optional, defaults to False):
                Guided Inpainting to Correct Image, it is preferable to use low values for strength.
            adetailer_params (Dict[str, Any], optional, defaults to {}):
                Placeholder for adetailer parameters.
            additional_prompt (str, optional):
                Placeholder for additional prompt.
            upscaler_model_path (str, optional):
                Placeholder for upscaler model path.
            upscaler_increases_size (float, optional, defaults to 1.5):
                Placeholder for upscaler increases size parameter.
            image (Any, optional):
                The image to be used for the Inpaint, ControlNet, or T2I adapter.
            preprocessor_name (str, optional, defaults to "None"):
                Preprocessor name for ControlNet.
            preprocess_resolution (int, optional, defaults to 512):
                Preprocess resolution for the Inpaint, ControlNet, or T2I adapter.
            image_resolution (int, optional, defaults to 512):
                Image resolution for the Inpaint, ControlNet, or T2I adapter.
            image_mask (Any, optional):
                Path image mask for the Inpaint.
            strength (float, optional, defaults to 1.0):
                Strength parameter for the Inpaint.
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
            generator_in_cpu (bool, optional, defaults to False):
                The generator by default is specified on the GPU. To obtain more consistent results across various environments, 
                it is preferable to use the generator on the CPU.
            leave_progress_bar (bool, optional, defaults to False):
                Leave the progress bar after generating the image.
            disable_progress_bar (bool, optional, defaults to False):
                Do not display the progress bar during image generation.
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
                
            Additional parameters that will be used in ControlNet depending on the task:
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
        
            Additional parameters that will be used in T2I adapter depending on the task:
                - image
                - preprocess_resolution
                - image_resolution
                - t2i_adapter_preprocessor
                - t2i_adapter_conditioning_scale
                - t2i_adapter_conditioning_factor
            
        """

        if self.task_name != "txt2img" and image == None:
            raise ValueError
        if img_height % 8 != 0 or img_width % 8 != 0:
            raise ValueError("Height and width must be divisible by 8")
        if control_guidance_start >= control_guidance_end:
            raise ValueError(
                "Control guidance start (ControlNet Start Threshold) cannot be larger or equal to control guidance end (ControlNet Stop Threshold)"
            )

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

        # in call
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
                    print(f"LoRA in memory: {single_lora}")
            pass

        else:
            # print("_un, re and load_")
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

        # FreeU
        if FreeU:
            print("FreeU active")
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

        # Prompt Optimizations for 1.5
        if self.class_name == "StableDiffusionPipeline":

            prompt_emb, negative_prompt_emb =  self.create_prompt_embeds(
                prompt = prompt,
                negative_prompt = negative_prompt,
                textual_inversion = textual_inversion,
                clip_skip = clip_skip,
                syntax_weights = syntax_weights,
            )

            # Prompt Optimizations for SDXL
        else:
            if self.embed_loaded != textual_inversion and textual_inversion != []:
                # implement
                print("SDXL textual inversion not available")

            # Clip skip
            if clip_skip:
                # clip_skip_diffusers = None #clip_skip - 1 # future update
                compel = Compel(
                    tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
                    text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True],
                    truncate_long_prompts=False,
                )
            else:
                # clip_skip_diffusers = None # clip_skip = None # future update
                compel = Compel(
                    tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
                    text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
                    requires_pooled=[False, True],
                    truncate_long_prompts=False,
                )

            # Prompt weights for textual inversion
            # prompt_ti = self.pipe.maybe_convert_prompt(prompt, self.pipe.tokenizer)
            # negative_prompt_ti = self.pipe.maybe_convert_prompt(negative_prompt, self.pipe.tokenizer)

            # prompt syntax style a1...
            if syntax_weights == "Classic":
                self.pipe.to("cuda")
                prompt_ti = get_embed_new(prompt, self.pipe, compel, only_convert_string=True)
                negative_prompt_ti = get_embed_new(negative_prompt, self.pipe, compel, only_convert_string=True)
            else:
                prompt_ti = prompt
                negative_prompt_ti = negative_prompt

            conditioning, pooled = compel([prompt_ti, negative_prompt_ti])
            prompt_emb = None
            negative_prompt_emb = None



        if torch.cuda.is_available() and xformers_memory_efficient_attention:
            if xformers_memory_efficient_attention:
                self.pipe.enable_xformers_memory_efficient_attention()
            else:
                self.pipe.disable_xformers_memory_efficient_attention()

        try:
            self.pipe.scheduler = self.get_scheduler(sampler)
        except:
            print("Error in sampler, please try again")
            self.pipe = None
            torch.cuda.empty_cache()
            gc.collect()
            return

        self.pipe.safety_checker = None

        # Get Control image
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
                self.pipe = None
                torch.cuda.empty_cache()
                gc.collect()
                if gui_active:
                    print(
                        "To use this function, you have to upload an image in the cell below first 👇"
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
                print("Unsupported image type")
                self.pipe = None
                torch.cuda.empty_cache()
                gc.collect()
                raise ValueError(
                    "Unsupported image type; Bug report to https://github.com/R3gm/stablepy or https://github.com/R3gm/SD_diffusers_interactive"
                )  # return

        # Get params preprocess
        preprocess_params_config = {}
        if self.task_name != "txt2img" and self.task_name != "inpaint":
            preprocess_params_config["image"] = array_rgb
            preprocess_params_config["image_resolution"] = image_resolution
            # preprocess_params_config["additional_prompt"] = additional_prompt # ""

            if self.task_name != "ip2p":
                if self.task_name != "shuffle":
                    preprocess_params_config[
                        "preprocess_resolution"
                    ] = preprocess_resolution
                if self.task_name != "mlsd" and self.task_name != "canny":
                    preprocess_params_config["preprocessor_name"] = preprocessor_name

        # RUN Preprocess sd
        if self.task_name == "inpaint":
            # Get mask for Inpaint
            if gui_active or os.path.exists(image_mask):
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
                print(f"Mask saved: {mask_control}")

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
            print("Openpose")
            control_image = self.process_openpose(**preprocess_params_config)

        elif self.task_name == "canny":
            print("Canny")
            control_image = self.process_canny(
                **preprocess_params_config,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
            )

        elif self.task_name == "mlsd":
            print("MLSD")
            control_image = self.process_mlsd(
                **preprocess_params_config,
                value_threshold=value_threshold,
                distance_threshold=distance_threshold,
            )

        elif self.task_name == "scribble":
            print("Scribble")
            control_image = self.process_scribble(**preprocess_params_config)

        elif self.task_name == "softedge":
            print("Softedge")
            control_image = self.process_softedge(**preprocess_params_config)

        elif self.task_name == "segmentation":
            print("Segmentation")
            control_image = self.process_segmentation(**preprocess_params_config)

        elif self.task_name == "depth":
            print("Depth")
            control_image = self.process_depth(**preprocess_params_config)

        elif self.task_name == "normalbae":
            print("NormalBae")
            control_image = self.process_normal(**preprocess_params_config)

        elif self.task_name == "lineart":
            print("Lineart")
            control_image = self.process_lineart(**preprocess_params_config)

        elif self.task_name == "shuffle":
            print("Shuffle")
            control_image = self.process_shuffle(**preprocess_params_config)

        elif self.task_name == "ip2p":
            print("Ip2p")
            control_image = self.process_ip2p(**preprocess_params_config)

        # RUN Preprocess sdxl
        if self.class_name == "StableDiffusionXLPipeline":
            # Get params preprocess XL
            preprocess_params_config_xl = {}
            if self.task_name != "txt2img" and self.task_name != "inpaint":
                preprocess_params_config_xl["image"] = array_rgb
                preprocess_params_config_xl["preprocess_resolution"] = preprocess_resolution
                preprocess_params_config_xl["image_resolution"] = image_resolution
                # preprocess_params_config_xl["additional_prompt"] = additional_prompt # ""

            if self.task_name == "sdxl_canny": # preprocessor true default
                print("SDXL Canny: Preprocessor active by default")
                control_image = self.process_canny(
                    **preprocess_params_config_xl,
                    low_threshold=low_threshold,
                    high_threshold=high_threshold,
                )
            elif self.task_name == "sdxl_openpose":
                print("SDXL Openpose")
                control_image = self.process_openpose(
                    preprocessor_name = "Openpose" if t2i_adapter_preprocessor else "None",
                    **preprocess_params_config_xl,
                )
            elif self.task_name == "sdxl_sketch":
                print("SDXL Scribble")
                control_image = self.process_scribble(
                    preprocessor_name = "PidiNet" if t2i_adapter_preprocessor else "None",
                    **preprocess_params_config_xl,
                )
            elif self.task_name == "sdxl_depth-midas":
                print("SDXL Depth")
                control_image = self.process_depth(
                    preprocessor_name = "Midas" if t2i_adapter_preprocessor else "None",
                    **preprocess_params_config_xl,
                )
            elif self.task_name == "sdxl_lineart":
                print("SDXL Lineart")
                control_image = self.process_lineart(
                    preprocessor_name = "Lineart" if t2i_adapter_preprocessor else "None",
                    **preprocess_params_config_xl,
                )

        # Get general params for TASK
        if self.class_name == "StableDiffusionPipeline":
            # Base params pipe sd
            pipe_params_config = {
                "prompt": None,  # prompt, #self.get_prompt(prompt, additional_prompt),
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
                print(f"Image resolution: {str(init_image.size)}")
            elif self.task_name != "txt2img" and self.task_name != "inpaint":
                pipe_params_config["image"] = control_image
                pipe_params_config["adapter_conditioning_scale"] = t2i_adapter_conditioning_scale
                pipe_params_config["adapter_conditioning_factor"] = t2i_adapter_conditioning_factor
                print(f"Image resolution: {str(control_image.size)}")
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
            print(f"Image resolution: {str(init_image.size)}")
        elif self.task_name != "txt2img" and self.task_name != "inpaint":
            pipe_params_config["control_image"] = control_image
            pipe_params_config[
                "controlnet_conditioning_scale"
            ] = controlnet_conditioning_scale
            pipe_params_config["control_guidance_start"] = control_guidance_start
            pipe_params_config["control_guidance_end"] = control_guidance_end
            print(f"Image resolution: {str(control_image.size)}")

        # Adetailer params and pipe
        if adetailer_active and self.class_name == "StableDiffusionPipeline":
            prompt_empty = (
                adetailer_params["inpaint_only"]["prompt"] is None
                or adetailer_params["inpaint_only"]["prompt"] == ""
            )
            negative_prompt_empty = (
                adetailer_params["inpaint_only"]["negative_prompt"] is None
                or adetailer_params["inpaint_only"]["negative_prompt"] == ""
            )

            if prompt_empty and negative_prompt_empty:
                adetailer_params["inpaint_only"]["prompt"] = None
                adetailer_params["inpaint_only"]["prompt_embeds"] = prompt_emb
                adetailer_params["inpaint_only"]["negative_prompt"] = None
                adetailer_params["inpaint_only"]["negative_prompt_embeds"] = negative_prompt_emb
            else:
                prompt_ad = (
                    prompt if prompt_empty else adetailer_params["inpaint_only"]["prompt"]
                )
                negative_prompt_ad = (
                    negative_prompt if negative_prompt_empty else adetailer_params["inpaint_only"]["negative_prompt"]
                )

                prompt_emb_ad, negative_prompt_emb_ad = self.create_prompt_embeds(
                    prompt=prompt_ad,
                    negative_prompt=negative_prompt_ad,
                    textual_inversion=textual_inversion,
                    clip_skip=clip_skip,
                    syntax_weights=syntax_weights,
                )

                adetailer_params["inpaint_only"]["prompt"] = None
                adetailer_params["inpaint_only"]["prompt_embeds"] = prompt_emb_ad
                adetailer_params["inpaint_only"]["negative_prompt"] = None
                adetailer_params["inpaint_only"]["negative_prompt_embeds"] = negative_prompt_emb_ad

            adetailer = AdCnPreloadPipe(self.pipe)  # use the loaded sampler
            adetailer.inpaint_pipeline.set_progress_bar_config(leave=leave_progress_bar)
            adetailer.inpaint_pipeline.set_progress_bar_config(
                disable=disable_progress_bar
            )

        ### RUN PIPE ###
        for i in range(loop_generation):
            calculate_seed = random.randint(0, 2147483647) if seed == -1 else seed
            if generator_in_cpu or self.device.type == "cpu":
                pipe_params_config["generator"] = torch.Generator().manual_seed(
                    calculate_seed
                )
            else:
                try:
                    pipe_params_config["generator"] = torch.Generator(
                        "cuda"
                    ).manual_seed(calculate_seed)
                except:
                    print("Generator in CPU")
                    pipe_params_config["generator"] = torch.Generator().manual_seed(
                        calculate_seed
                    )

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
                if self.task_name != "txt2img" and self.task_name != "inpaint":
                    images = [control_image] + images
            elif self.task_name == "txt2img":
                images = self.run_pipe_SD(**pipe_params_config)
            elif self.task_name == "inpaint":
                images = self.run_pipe_inpaint(**pipe_params_config)
            elif self.task_name != "txt2img" and self.task_name != "inpaint":
                results = self.run_pipe(
                    **pipe_params_config
                )  ## pipe ControlNet add condition_weights
                images = [control_image] + results
                del results

            torch.cuda.empty_cache()
            gc.collect()

            # Adetailer stuff
            if adetailer_active and self.class_name == "StableDiffusionPipeline":
                # image_pil_list = []
                # for img_single in images:
                # image_ad = img_single.convert("RGB")
                # image_pil_list.append(image_ad)
                if self.task_name != "txt2img" and self.task_name != "inpaint":
                    images = images[1:]
                images = ad_model_process(
                    adetailer=adetailer,
                    image_list_task=images,
                    **adetailer_params,
                )
                if self.task_name != "txt2img" and self.task_name != "inpaint":
                    images = [control_image] + images
                # del adetailer
                torch.cuda.empty_cache()
                gc.collect()

            # Upscale
            if upscaler_model_path != None:
                scaler = UpscalerESRGAN()
                result_scaler = []
                for img_pre_up in images:
                    image_pos_up = scaler.upscale(
                        img_pre_up, upscaler_increases_size, upscaler_model_path
                    )
                    torch.cuda.empty_cache()
                    gc.collect()
                    result_scaler.append(image_pos_up)
                images = result_scaler

            # Show images if loop
            if loop_generation > 1:
                mediapy.show_images(images)
                # print(image_list)
                # del images
                time.sleep(0.5)

            # Save images
            image_list = []
            metadata = [
                prompt,
                negative_prompt,
                self.base_model_id,
                self.vae_model,
                num_steps,
                guidance_scale,
                sampler,
                calculate_seed,
                img_width,
                img_height,
                clip_skip,
            ]
            for image_ in images:
                image_path = save_pil_image_with_metadata(image_, "./images", metadata)
                image_list.append(image_path)

            if loop_generation > 1:
                torch.cuda.empty_cache()
                gc.collect()
                print(image_list)
            print(f"Seed:\n{calculate_seed}")

        torch.cuda.empty_cache()
        gc.collect()

        return images, image_list
