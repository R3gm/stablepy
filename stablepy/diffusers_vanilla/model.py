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
)
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
from .prompt_weights import prompt_weight_conversor, tokenize_line, merge_embeds
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
    "Openpose": "lllyasviel/control_v11p_sd15_openpose",
    "Canny": "lllyasviel/control_v11p_sd15_canny",
    "MLSD": "lllyasviel/control_v11p_sd15_mlsd",
    "scribble": "lllyasviel/control_v11p_sd15_scribble",
    "softedge": "lllyasviel/control_v11p_sd15_softedge",
    "segmentation": "lllyasviel/control_v11p_sd15_seg",
    "depth": "lllyasviel/control_v11f1p_sd15_depth",
    "NormalBae": "lllyasviel/control_v11p_sd15_normalbae",
    "lineart": "lllyasviel/control_v11p_sd15_lineart",
    "lineart_anime": "lllyasviel/control_v11p_sd15s2_lineart_anime",
    "shuffle": "lllyasviel/control_v11e_sd15_shuffle",
    "ip2p": "lllyasviel/control_v11e_sd15_ip2p",
    "Inpaint": "lllyasviel/control_v11p_sd15_inpaint",
    "txt2img": "nothinghere",
}


def download_all_controlnet_weights() -> None:
    for model_id in CONTROLNET_MODEL_IDS.values():
        ControlNetModel.from_pretrained(model_id)


class Model_Diffusers:
    def __init__(
        self,
        base_model_id: str = "runwayml/stable-diffusion-v1-5",
        task_name: str = "Canny",
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

        self.pipe = self.load_pipe(
            base_model_id, task_name, vae_model, type_model_precision
        )
        self.preprocessor = Preprocessor()

    def load_pipe(
        self,
        base_model_id: str,
        task_name,
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
            and type_model_precision == self.type_model_precision
        ):
            print("Previous loaded")
            return self.pipe
        if (
            base_model_id == self.base_model_id
            and task_name == self.task_name
            and hasattr(self, "pipe")
            and self.vae_model == vae_model
            and self.pipe is not None
            and reload == False
            and self.device == "cpu"
        ):
            print("Pipe in CPU")
            return self.pipe

        self.type_model_precision = (
            type_model_precision if torch.cuda.is_available() else torch.float32
        )
        self.pipe = None
        torch.cuda.empty_cache()
        gc.collect()

        model_id = CONTROLNET_MODEL_IDS[task_name]

        if task_name == "txt2img":
            if os.path.exists(base_model_id):
                if self.type_model_precision == torch.float32:
                    print("Working with full precision")
                pipe = StableDiffusionPipeline.from_single_file(
                    base_model_id,
                    vae=None
                    if vae_model == None
                    else AutoencoderKL.from_single_file(
                        vae_model
                    ),  # , torch_dtype=self.type_model_precision
                    torch_dtype=self.type_model_precision,
                )
                pipe.safety_checker = None
            else:
                print("Default VAE: madebyollin/sdxl-vae-fp16-fix")
                pipe = DiffusionPipeline.from_pretrained(
                    base_model_id,
                    vae=AutoencoderKL.from_pretrained(
                        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
                    ),
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
                )
                pipe.safety_checker = None
            print("Loaded txt2img pipeline")
        elif task_name == "Inpaint":
            if self.type_model_precision == torch.float32:
                print("Working with full precision")
            controlnet = ControlNetModel.from_pretrained(
                model_id, torch_dtype=self.type_model_precision
            )
            if os.path.exists(base_model_id):
                pipe = StableDiffusionControlNetInpaintPipeline.from_single_file(
                    base_model_id,
                    vae=None
                    if vae_model == None
                    else AutoencoderKL.from_single_file(vae_model),
                    safety_checker=None,
                    controlnet=controlnet,
                    torch_dtype=self.type_model_precision,
                )
            print("Loaded ControlNet Inpaint pipeline")
        else:
            if self.type_model_precision == torch.float32:
                print("Working with full precision")
            controlnet = ControlNetModel.from_pretrained(
                model_id, torch_dtype=self.type_model_precision
            )  # for all
            if os.path.exists(base_model_id):
                pipe = StableDiffusionControlNetPipeline.from_single_file(
                    base_model_id,
                    vae=None
                    if vae_model == None
                    else AutoencoderKL.from_single_file(vae_model),
                    safety_checker=None,
                    controlnet=controlnet,
                    torch_dtype=self.type_model_precision,
                )
            else:
                raise ZeroDivisionError("Not implemented for SDXL")
            #     pipe = StableDiffusionControlNetPipeline.from_pretrained(
            #         base_model_id,
            #         vae = AutoencoderKL.from_pretrained(base_model_id, subfolder='vae') if vae_model == None else AutoencoderKL.from_single_file(vae_model),
            #         safety_checker=None,
            #         controlnet=controlnet,
            #         torch_dtype=torch.float16)
            # print('Loaded ControlNet pipeline')

            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        if self.device.type == "cuda":
            pipe.enable_xformers_memory_efficient_attention()

        pipe.to(self.device)
        torch.cuda.empty_cache()
        gc.collect()
        self.pipe = pipe
        self.base_model_id = base_model_id
        self.task_name = task_name
        self.vae_model = vae_model

        self.lora_memory = [None, None, None, None, None]  # no need __init__
        self.lora_scale_memory = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.FreeU = False
        self.embed_loaded = []
        return pipe

    def set_base_model(self, base_model_id: str) -> str:
        if not base_model_id or base_model_id == self.base_model_id:
            return self.base_model_id
        del self.pipe
        torch.cuda.empty_cache()
        gc.collect()
        try:
            self.pipe = self.load_pipe(base_model_id, self.task_name, self.vae_model)
        except Exception:
            self.pipe = self.load_pipe(
                self.base_model_id, self.task_name, self.vae_model
            )
        return self.base_model_id

    def load_controlnet_weight(self, task_name: str) -> None:
        if task_name == self.task_name:
            return
        if self.pipe is not None and hasattr(self.pipe, "controlnet"):
            del self.pipe.controlnet
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
        self.task_name = task_name

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

        self.load_controlnet_weight("Canny")

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
        self.load_controlnet_weight("MLSD")

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
        self.load_controlnet_weight("scribble")

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

        self.load_controlnet_weight("scribble")

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
        self.load_controlnet_weight("softedge")

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
        self.load_controlnet_weight("Openpose")

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
        self.load_controlnet_weight("segmentation")

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
        self.load_controlnet_weight("depth")

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
        self.load_controlnet_weight("NormalBae")

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
        if "anime" in preprocessor_name:
            self.load_controlnet_weight("lineart_anime")
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
        self.load_controlnet_weight("shuffle")

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
        self.load_controlnet_weight("ip2p")

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

        self.load_controlnet_weight("Inpaint")

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

            case "DDIMScheduler":
                return DDIMScheduler.from_config(self.pipe.scheduler.config)

            case "DEISMultistepScheduler":
                return DEISMultistepScheduler.from_config(self.pipe.scheduler.config)

            case "UniPCMultistepScheduler":
                return UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

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
        prompt=None,
        negative_prompt="",
        # prompt_embeds = None,
        # negative_prompt_embeds = None,
        img_height=512,
        img_width=512,
        num_images=1,
        num_steps=30,
        guidance_scale=7.5,
        clip_skip=True,
        seed=-1,
        image=None,  # path, np.array, or PIL image
        preprocessor_name=None,
        preprocess_resolution=512,
        image_resolution=512,
        additional_prompt="",
        image_mask=None,
        strength=1.0,
        low_threshold=100,  # Canny
        high_threshold=200,  # Canny
        value_threshold=0.1,  # MLSD
        distance_threshold=0.1,  # MLSD
        lora_A=None,
        lora_scale_A=1.0,
        lora_B=None,
        lora_scale_B=1.0,
        lora_C=None,
        lora_scale_C=1.0,
        lora_D=None,
        lora_scale_D=1.0,
        lora_E=None,
        lora_scale_E=1.0,
        active_textual_inversion=False,
        textual_inversion=[],  # List of tuples [(activation_token, path_embedding),...]
        convert_weights_prompt=False,
        sampler="DPM++ 2M",
        xformers_memory_efficient_attention=True,
        gui_active=False,
        loop_generation=1,
        controlnet_conditioning_scale=1.0,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
        generator_in_cpu=False,  # Initial noise not in CPU
        FreeU=False,
        adetailer_active=False,
        adetailer_params={},
        leave_progress_bar=False,
        disable_progress_bar=False,
        image_previews=False,
        upscaler_model_path=None,
        upscaler_increases_size=1.5,
    ):
        if self.task_name != "txt2img" and image == None:
            raise ValueError
        if img_height % 8 != 0 or img_width % 8 != 0:
            raise ValueError("Height and width must be divisible by 8")
        if control_guidance_start >= control_guidance_end:
            raise ValueError(
                "Control guidance start (ControlNet Start Threshold) cannot be larger or equal to control guidance end (ControlNet Stop Threshold)"
            )

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

        # self.pipe.to(self.device)
        # self.pipe.unfuse_lora()
        # self.pipe.unload_lora_weights()
        # self.pipe = self.process_lora(lora_A, lora_scale_A)
        # self.pipe = self.process_lora(lora_B, lora_scale_B)
        # self.pipe = self.process_lora(lora_C, lora_scale_C)
        # self.pipe = self.process_lora(lora_D, lora_scale_D)
        # self.pipe = self.process_lora(lora_E, lora_scale_E)

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
                    print(f"LoRA in memory:{single_lora}")
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
            if os.path.exists(self.base_model_id):
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
        if os.path.exists(self.base_model_id):
            if active_textual_inversion and self.embed_loaded != textual_inversion:
                # Textual Inversion
                for name, directory_name in textual_inversion:
                    try:
                        if directory_name.endswith(".pt"):
                            model = torch.load(directory_name, map_location=self.device)
                            model_tensors = model.get("string_to_param").get("*")
                            s_model = {"emb_params": model_tensors}
                            # if directory_name.endswith('.pt'):
                            #     new_file_path = directory_name[:-3] + '.safetensors'
                            # else:
                            #     new_file_path = directory_name + '.safetensors'
                            # save_file(s_model, new_file_path)
                            self.pipe.load_textual_inversion(s_model, token=name)

                        else:
                            # self.pipe.text_encoder.resize_token_embeddings(len(self.pipe.tokenizer),pad_to_multiple_of=128)
                            # self.pipe.load_textual_inversion("./bad_prompt.pt", token="baddd")
                            self.pipe.load_textual_inversion(directory_name, token=name)
                        if not gui_active:
                            print(f"Applied : {name}")
                        self.embed_loaded = textual_inversion
                    except ValueError:
                        print(f"Previous loaded embed {name}")
                        pass
                    except:
                        print(f"Can't apply embed {name}")

            # Clip skip
            if clip_skip:
                # clip_skip_diffusers = None #clip_skip - 1 # future update
                compel = Compel(
                    tokenizer=self.pipe.tokenizer,
                    text_encoder=self.pipe.text_encoder,
                    truncate_long_prompts=False,
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED,
                )
            else:
                # clip_skip_diffusers = None # clip_skip = None # future update
                compel = Compel(
                    tokenizer=self.pipe.tokenizer,
                    text_encoder=self.pipe.text_encoder,
                    truncate_long_prompts=False,
                )

            # Prompt weights for textual inversion
            prompt_ti = self.pipe.maybe_convert_prompt(prompt, self.pipe.tokenizer)
            negative_prompt_ti = self.pipe.maybe_convert_prompt(
                negative_prompt, self.pipe.tokenizer
            )

            # prompt syntax style a1...
            if convert_weights_prompt:
                prompt_ti = prompt_weight_conversor(prompt_ti)
                negative_prompt_ti = prompt_weight_conversor(negative_prompt_ti)

            # prompt embed chunks style a1...
            prompt_emb = merge_embeds(
                tokenize_line(prompt_ti, self.pipe.tokenizer), compel
            )
            negative_prompt_emb = merge_embeds(
                tokenize_line(negative_prompt_ti, self.pipe.tokenizer), compel
            )

            # fix error shape
            if prompt_emb.shape != negative_prompt_emb.shape:
                (
                    prompt_emb,
                    negative_prompt_emb,
                ) = compel.pad_conditioning_tensors_to_same_length(
                    [prompt_emb, negative_prompt_emb]
                )

            compel = None
            del compel

        # Prompt Optimizations for SDXL
        else:
            if active_textual_inversion:
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
            if convert_weights_prompt:
                prompt_ti = prompt_weight_conversor(prompt)
                negative_prompt_ti = prompt_weight_conversor(negative_prompt)
            else:
                prompt_ti = prompt
                negative_prompt_ti = negative_prompt

            conditioning, pooled = compel([prompt_ti, negative_prompt_ti])
            prompt_emb = None
            negative_prompt_emb = None

            compel = None
            del compel

        if torch.cuda.is_available():
            if xformers_memory_efficient_attention:
                self.pipe.enable_xformers_memory_efficient_attention()
            else:
                self.pipe.disable_xformers_memory_efficient_attention()

        try:
            self.pipe.scheduler = self.get_scheduler(sampler)
        except:
            print(
                "Error in sampler, please try again; Bug report to https://github.com/R3gm/stablepy or https://github.com/R3gm/SD_diffusers_interactive"
            )
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
        if self.task_name != "txt2img" and self.task_name != "Inpaint":
            preprocess_params_config["image"] = array_rgb
            preprocess_params_config["image_resolution"] = image_resolution
            # preprocess_params_config["additional_prompt"] = additional_prompt # ""

            if self.task_name != "ip2p":
                if self.task_name != "shuffle":
                    preprocess_params_config[
                        "preprocess_resolution"
                    ] = preprocess_resolution
                if self.task_name != "MLSD" and self.task_name != "Canny":
                    preprocess_params_config["preprocessor_name"] = preprocessor_name

        # RUN Preprocess
        if self.task_name == "Inpaint":
            # Get mask for Inpaint
            if gui_active or os.path.exists(image_mask):
                # Read image mask from gui
                mask_control_img = Image.open(image_mask)
                numpy_array_mask = np.array(mask_control_img, dtype=np.uint8)
                array_rgb_mask = numpy_array_mask[:, :, :3]
            elif not gui_active:
                # Convert control image for draw
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

        elif self.task_name == "Openpose":
            print("Openpose")
            control_image = self.process_openpose(**preprocess_params_config)

        elif self.task_name == "Canny":
            print("Canny")
            control_image = self.process_canny(
                **preprocess_params_config,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
            )

        elif self.task_name == "MLSD":
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

        elif self.task_name == "NormalBae":
            print("NormalBae")
            control_image = self.process_normal(**preprocess_params_config)

        elif "lineart" in self.task_name:
            print("Lineart")
            control_image = self.process_lineart(**preprocess_params_config)

        elif self.task_name == "shuffle":
            print("Shuffle")
            control_image = self.process_shuffle(**preprocess_params_config)

        elif self.task_name == "ip2p":
            print("Ip2p")
            control_image = self.process_ip2p(**preprocess_params_config)

        # Get params for TASK
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

        if not os.path.exists(self.base_model_id):
            # pipe_params_config["prompt_embeds"]                 = conditioning[0:1]
            # pipe_params_config["pooled_prompt_embeds"]          = pooled[0:1],
            # pipe_params_config["negative_prompt_embeds"]        = conditioning[1:2]
            # pipe_params_config["negative_pooled_prompt_embeds"] = pooled[1:2],
            # pipe_params_config["conditioning"]                  = conditioning,
            # pipe_params_config["pooled"]                        = pooled,
            # pipe_params_config["height"]                        = img_height
            # pipe_params_config["width"]                         = img_width
            pass
        elif self.task_name == "txt2img":
            pipe_params_config["height"] = img_height
            pipe_params_config["width"] = img_width
        elif self.task_name == "Inpaint":
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
        elif self.task_name != "txt2img" and self.task_name != "Inpaint":
            pipe_params_config["control_image"] = control_image
            pipe_params_config[
                "controlnet_conditioning_scale"
            ] = controlnet_conditioning_scale
            pipe_params_config["control_guidance_start"] = control_guidance_start
            pipe_params_config["control_guidance_end"] = control_guidance_end
            print(f"Image resolution: {str(control_image.size)}")

        # Adetailer params and pipe
        if adetailer_active and os.path.exists(self.base_model_id):
            # Check if "prompt" is empty or None in adetailer_params["inpaint_only"]
            if (
                adetailer_params["inpaint_only"]["prompt"] is None
                or adetailer_params["inpaint_only"]["prompt"] == ""
            ):
                adetailer_params["inpaint_only"]["prompt"] = None
                adetailer_params["inpaint_only"]["prompt_embeds"] = prompt_emb
            if (
                adetailer_params["inpaint_only"]["negative_prompt"] is None
                or adetailer_params["inpaint_only"]["negative_prompt"] == ""
            ):
                adetailer_params["inpaint_only"]["negative_prompt"] = None
                adetailer_params["inpaint_only"][
                    "negative_prompt_embeds"
                ] = negative_prompt_emb
            adetailer = AdCnPreloadPipe(self.pipe)  # use the loaded sampler
            adetailer.inpaint_pipeline.set_progress_bar_config(leave=leave_progress_bar)
            adetailer.inpaint_pipeline.set_progress_bar_config(
                disable=disable_progress_bar
            )

        ### RUN PIPE ###
        for i in range(loop_generation):
            calculate_seed = random.randint(0, 2147483647) if seed == -1 else seed
            if generator_in_cpu:
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

            if not os.path.exists(self.base_model_id):
                # images = self.run_pipe_SDXL(**pipe_params_config)
                images = self.pipe(
                    prompt=None,
                    negative_prompt=None,
                    prompt_embeds=conditioning[0:1],
                    pooled_prompt_embeds=pooled[0:1],
                    negative_prompt_embeds=conditioning[1:2],
                    negative_pooled_prompt_embeds=pooled[1:2],
                    height=img_height,
                    width=img_width,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    clip_skip=None,
                    num_images_per_prompt=num_images,
                    generator=pipe_params_config["generator"],
                ).images
            elif self.task_name == "txt2img":
                images = self.run_pipe_SD(**pipe_params_config)
            elif self.task_name == "Inpaint":
                images = self.run_pipe_inpaint(**pipe_params_config)
            elif self.task_name != "txt2img" and self.task_name != "Inpaint":
                results = self.run_pipe(
                    **pipe_params_config
                )  ## pipe ControlNet add condition_weights
                images = [control_image] + results
                del results

            torch.cuda.empty_cache()
            gc.collect()

            # Adetailer stuff
            if adetailer_active and os.path.exists(self.base_model_id):
                # image_pil_list = []
                # for img_single in images:
                # image_ad = img_single.convert("RGB")
                # image_pil_list.append(image_ad)
                if self.task_name != "txt2img" and self.task_name != "Inpaint":
                    images = images[1:]
                images = ad_model_process(
                    adetailer=adetailer,
                    image_list_task=images,
                    **adetailer_params,
                )
                if self.task_name != "txt2img" and self.task_name != "Inpaint":
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
                time.sleep(1)

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

        # if select_lora1.value != "None":
        #     model.pipe.unfuse_lora()
        #     model.pipe.unload_lora_weights()
        # if select_lora2.value != "None" or select_lora3.value != "None":
        #     print('BETA: reload weights for lora')
        #     model.load_pipe(select_model.value, task_name=options_controlnet.value, vae_model = vae_model_dropdown.value, reload=True)
        # self.pipe.to(self.device)
        # self.pipe = self.process_lora(lora_A, lora_scale_A, unload=True)
        # self.pipe = self.process_lora(lora_B, lora_scale_B, unload=True)
        # self.pipe = self.process_lora(lora_C, lora_scale_C, unload=True)
        # self.pipe = self.process_lora(lora_D, lora_scale_D, unload=True)
        # self.pipe = self.process_lora(lora_E, lora_scale_E, unload=True)
        # model.pipe.unfuse_lora()
        # model.pipe.unload_lora_weights()

        torch.cuda.empty_cache()
        gc.collect()

        return images, image_list
