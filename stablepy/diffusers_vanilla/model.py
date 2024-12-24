import time
import os
import copy
import warnings
import traceback
import threading
import json

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
    FluxTransformer2DModel,
)
from accelerate import init_on_device
from huggingface_hub import hf_hub_download
import torch
import random
import cv2
from diffusers import (
    DDIMScheduler,
)
from .constants import (
    SD15,
    SDXL,
    FLUX,
    CLASS_DIFFUSERS_TASK,
    CLASS_PAG_DIFFUSERS_TASK,
    CONTROLNET_MODEL_IDS,
    FLUX_CN_UNION_MODES,
    FLASH_LORA,
    SCHEDULER_CONFIG_MAP,
    scheduler_names,
    IP_ADAPTER_MODELS,
    REPO_IMAGE_ENCODER,
    FLASH_AUTO_LOAD_SAMPLER,
    SDXL_CN_UNION_PROMAX_MODES,
)
from diffusers.utils import load_image
from .prompt_weights import add_comma_after_pattern_ti
from .utils import (
    save_pil_image_with_metadata,
    checkpoint_model_type,
    get_string_metadata,
    extra_string_metadata,
    assemble_filename_pattern,
    process_prompts_valid,
    convert_image_to_numpy_array,
    latents_to_rgb,
    load_cn_diffusers,
    check_variant_file,
    cachebox,
    release_resources,
    validate_and_update_params,
    get_seeds,
    get_flux_components_info,
    CURRENT_TASK_PARAMS,
)
from .lora_loader import lora_mix_load, load_no_fused_lora
from .inpainting_canvas import draw, make_inpaint_condition
from .adetailer import ad_model_process
from ..logging.logging_setup import logger
from .extra_model_loaders import custom_task_model_loader
from .high_resolution import process_images_high_resolution, LATENT_UPSCALERS
from .style_prompt_config import (
    styles_data,
    STYLE_NAMES,
    get_json_content,
    apply_style,
)
import mediapy
from PIL import Image
from typing import Union, Optional, List, Tuple, Dict, Any, Callable # noqa
import logging
import diffusers
from .main_prompt_embeds import (
    Promt_Embedder_SD1,
    Promt_Embedder_SDXL,
    Promt_Embedder_FLUX,
)
from .sampler_scheduler_config import (
    configure_scheduler,
    verify_schedule_integrity,
    check_scheduler_compatibility,
    ays_timesteps,
)
from .preprocessor.main_preprocessor import (
    Preprocessor,
    process_basic_task,
    get_preprocessor_params,
)
from .preprocessor.constans_preprocessor import T2I_PREPROCESSOR_NAME
from ..face_restoration.main_face_restoration import batch_process_face_restoration

logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
diffusers.utils.logging.set_verbosity(40)
warnings.filterwarnings(action="ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings(action="ignore", category=FutureWarning, module="transformers")


class PreviewGenerator:
    def __init__(self, *args, **kwargs):
        self.image_step = None
        self.fail_work = None
        self.stream_config(5, 8, False)

    def stream_config(self, concurrency=5, latent_resize_by=8, vae_decoding=False):
        """
        Configures the streaming settings for the model.

        Args:
            concurrency (int): Controls how often the preview images are generated and displayed in relation to the steps.
                               For example, a value of 2 displays an image every 2 steps. Default is 5.
            latent_resize_by (int): Controls the scaling size of the latent images. A value of 1 is useful for achieving
                                    high performance. Default is 8.
            vae_decoding (bool): Use the VAE to decode the preview images. If set to True, it may negatively impact
                                 performance. Default is False.
        """
        self.concurrency = concurrency
        self.latent_resize_by = latent_resize_by
        self.vae_decoding = vae_decoding

    def decode_tensors(self, pipe, step, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        if step % self.concurrency == 0:  # every how many steps
            logger.debug(step)
            if self.class_name == FLUX:
                latents = self.pipe._unpack_latents(latents, self.metadata[9], self.metadata[8], self.pipe.vae_scale_factor)
                latents = (latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
                batch_size, channels, height, width = latents.shape
                target_channels = 4
                group_size = channels // target_channels
                latents = latents.view(batch_size, target_channels, group_size, height, width).mean(dim=2)

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
    """
    Model_Diffusers class for generating images using diffusers.

    Args:
        base_model_id (str): ID of the base model. Default is "Lykon/dreamshaper-8".
        task_name (str): Name of the task. Default is "txt2img".
        vae_model: The VAE model to use. Default is None. To use the BakedVAE in the model, use the same base_model_id here.
        type_model_precision: Precision type for the model. Default is torch.float16.
        retain_task_model_in_cache (bool): Whether to retain the task model in cache. Default is False.
        device: Device to use for computation. Default is None (Automatic).
        controlnet_model (str): ControlNet model to use. Default is "Automatic".
        env_components (dict): Environment components. Default is None.
    """
    def __init__(
        self,
        base_model_id: str = "Lykon/dreamshaper-8",
        task_name: str = "txt2img",
        vae_model=None,
        type_model_precision=torch.float16,
        retain_task_model_in_cache=False,
        device=None,
        controlnet_model="Automatic",
        env_components=None,
    ):
        super().__init__()

        if isinstance(env_components, dict):
            self.env_components = env_components
            self.env_components.pop("transformer", None)
        else:
            self.env_components = None

        self.device = (
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )
        self.base_model_id = ""
        self.task_name = ""
        self.model_id_task = "Automatic"
        self.vae_model = None
        self.type_model_precision = (
            type_model_precision if torch.cuda.is_available() else torch.float32
        )  # For SD 1.5

        self.image_encoder_name = None
        self.image_encoder_module = None
        self.ip_adapter_config = None

        self.last_lora_error = ""
        self.num_loras = 7

        reload = False
        self.load_pipe(
            base_model_id,
            task_name,
            vae_model,
            type_model_precision,
            reload,
            retain_task_model_in_cache,
            controlnet_model,
        )
        self.preprocessor = Preprocessor()
        self.advanced_params()

        self.styles_data = styles_data
        self.STYLE_NAMES = STYLE_NAMES
        self.style_json_file = ""

    def advanced_params(
        self,
        image_preprocessor_cuda_active: bool = False,
    ):
        self.image_preprocessor_cuda_active = image_preprocessor_cuda_active

    def switch_pipe_class(
        self,
        class_name,
        task_name,
        model_id,
        enable_pag,
        verbose_info=False,
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
            scheduler=self.pipe.scheduler,
        )

        if class_name == FLUX:
            model_components["text_encoder_2"] = self.pipe.text_encoder_2
            model_components["tokenizer_2"] = self.pipe.tokenizer_2
            model_components["transformer"] = self.pipe.transformer

            if task_name == "txt2img":
                from diffusers import FluxPipeline
                self.pipe = FluxPipeline(**model_components)
            elif task_name in ["repaint", "inpaint"]:
                from .extra_pipe.flux.pipeline_flux_inpaint import FluxInpaintPipeline
                self.pipe = FluxInpaintPipeline(**model_components)
            elif task_name == "img2img":
                from .extra_pipe.flux.pipeline_flux_img2img import FluxImg2ImgPipeline
                self.pipe = FluxImg2ImgPipeline(**model_components)
            else:
                from .extra_pipe.flux.pipeline_flux_controlnet import FluxControlNetPipeline
                from .extra_pipe.flux.controlnet_flux import FluxControlNetModel

                if verbose_info:
                    logger.info(f"ControlNet model: {model_id}")

                if hasattr(self.pipe, "controlnet"):
                    model_components["controlnet"] = self.pipe.controlnet
                else:
                    model_components["controlnet"] = FluxControlNetModel.from_pretrained(
                        model_id,
                        torch_dtype=torch.bfloat16
                    )

                self.pipe = FluxControlNetPipeline(
                    **model_components
                )
            return None

        else:
            model_components["unet"] = self.pipe.unet
            model_components["feature_extractor"] = self.pipe.feature_extractor
            model_components["image_encoder"] = self.pipe.image_encoder

        if class_name == SD15:
            model_components["safety_checker"] = self.pipe.safety_checker
            model_components["requires_safety_checker"] = self.pipe.config.requires_safety_checker

            if task_name not in ["txt2img", "img2img"]:
                release_resources()

                if verbose_info:
                    logger.info(f"ControlNet model: {model_id}")
                if os.path.exists(model_id):
                    model_components["controlnet"] = load_cn_diffusers(
                        model_id,
                        "lllyasviel/control_v11p_sd15s2_lineart_anime",
                        self.type_model_precision,
                    )
                else:
                    model_components["controlnet"] = ControlNetModel.from_pretrained(
                        model_id, torch_dtype=self.type_model_precision
                    )
                tk = "controlnet"

                if task_name == "repaint":
                    tk = "inpaint"

        elif class_name == SDXL:
            model_components["text_encoder_2"] = self.pipe.text_encoder_2
            model_components["tokenizer_2"] = self.pipe.tokenizer_2

            if task_name not in ["txt2img", "inpaint", "img2img"]:
                if verbose_info:
                    logger.info(f"Task model: {model_id}")
                release_resources()

                # with init_on_device(self.device):

                if "t2i" not in task_name:

                    tk = "controlnet"

                    if os.path.exists(model_id):
                        if (
                            hasattr(self.pipe, "controlnet")
                            and hasattr(self.pipe.controlnet, "config")
                            and self.pipe.controlnet.config._name_or_path == model_id
                        ):
                            model_components["controlnet"] = self.pipe.controlnet
                        else:
                            if hasattr(self.pipe, "controlnet"):
                                # self.pipe.controlnet.to_empty(device=self.device)
                                self.pipe.__delattr__("controlnet")
                                self.pipe.controlnet = None
                                self.model_memory = {}
                                release_resources()
                            model_components["controlnet"] = load_cn_diffusers(
                                model_id,
                                "r3gm/controlnet-lineart-anime-sdxl-fp16",
                                torch.float16,
                            ).to(self.device)
                    else:

                        cls_controlnet = ControlNetModel

                        if all(kw in model_id.lower() for kw in ["union", "promax"]):
                            from .extra_pipe.sdxl.controlnet_union import ControlNetUnionModel
                            cls_controlnet = ControlNetUnionModel
                            tk = "controlnet_union+"

                        if (
                            hasattr(self.pipe, "controlnet")
                            and hasattr(self.pipe.controlnet, "config")
                            and self.pipe.controlnet.config._name_or_path == model_id
                            and self.pipe.controlnet.__class__.__name__ == cls_controlnet.__name__
                        ):
                            model_components["controlnet"] = self.pipe.controlnet
                        else:
                            if hasattr(self.pipe, "controlnet"):
                                # self.pipe.controlnet.to_empty(device=self.device)
                                self.pipe.__delattr__("controlnet")
                                self.pipe.controlnet = None
                                self.model_memory = {}
                                release_resources()
                            model_components["controlnet"] = cls_controlnet.from_pretrained(
                                model_id, torch_dtype=torch.float16, variant=check_variant_file(model_id, "fp16")
                            ).to(self.device)

                        if task_name == "repaint":
                            tk += "_inpaint"

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
                task_name == "shuffle" or
                (tk not in CLASS_PAG_DIFFUSERS_TASK[class_name])
                # (task_name in ["inpaint", "img2img"] and "XL" not in class_name)
            ):
                logger.warning(
                    f"PAG is not enabled for {class_name}-{tk} with {task_name}."
                )
                enable_pag = False

        # Load Pipeline
        # with init_on_device(self.device):
        if enable_pag:
            model_components["pag_applied_layers"] = "mid"
            self.pipe = CLASS_PAG_DIFFUSERS_TASK[class_name][tk](**model_components).to(self.device)
        else:
            self.pipe = CLASS_DIFFUSERS_TASK[class_name][tk](**model_components).to(self.device)

        if task_name == "img2img":
            self.pipe = AutoPipelineForImage2Image.from_pipe(self.pipe, enable_pag=enable_pag)

        # Create new base values
        self.pipe.to(self.device)
        release_resources()

    def load_pipe(
        self,
        base_model_id: str,
        task_name="txt2img",
        vae_model=None,
        type_model_precision=torch.float16,
        reload=False,
        retain_task_model_in_cache=False,
        controlnet_model="Automatic",
    ) -> DiffusionPipeline:
        """
        Load a diffusion pipeline model.
        Args:
            base_model_id (str): The ID of the base model to load.
            task_name (str, optional): The task name for the model. Defaults to "txt2img".
            vae_model (str, optional): The VAE model to use. Defaults to None. To use the BakedVAE in the model, use the same base_model_id here.
            type_model_precision (torch.dtype, optional): The precision type for the model. Defaults to torch.float16.
            reload (bool, optional): Whether to reload the model even if it is already loaded. Defaults to False.
            retain_task_model_in_cache (bool, optional): Whether to retain the task model in cache. Defaults to False.
            controlnet_model (str, optional): The controlnet model to use. Defaults to "Automatic".
        Returns:
            DiffusionPipeline: The loaded diffusion pipeline model.
        """

        if not controlnet_model or task_name in ["txt2img", "img2img"]:
            controlnet_model = "Automatic"

        if (
            base_model_id == self.base_model_id
            and task_name == self.task_name
            and controlnet_model == self.model_id_task
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
            self.image_encoder_module = None
            self.image_encoder_name = None
            if self.ip_adapter_config:
                self.pipe.unload_ip_adapter()
                self.ip_adapter_config = None
            release_resources()

            if hasattr(self, "pipe") and self.pipe is not None:
                for k_com, v_com in self.pipe.components.items():
                    if hasattr(v_com, "to_empty"):

                        if "Flux" in self.pipe.__class__.__name__ and isinstance(self.env_components, dict) and k_com in self.env_components:
                            pass
                        else:
                            v_com.to_empty(device=self.device)
                            self.pipe.__delattr__(k_com)
                        setattr(self.pipe, k_com, None)

            self.pipe = None
            # release_resources()
            self.task_name = ""
            self.model_memory = {}
            self.lora_memory = [None] * self.num_loras
            self.lora_scale_memory = [1.0] * self.num_loras
            self.lora_status = [None] * self.num_loras
            self.flash_config = None
            self.ip_adapter_config = None
            self.embed_loaded = []
            self.FreeU = False
            self.create_prompt_embeds.memory.clear()
            release_resources()

            # Load new model
            if os.path.isfile(base_model_id):  # exists or not same # if os.path.exists(base_model_id):

                if base_model_id.endswith(".safetensors"):
                    model_type, sampling_type, scheduler_config, has_baked_vae = checkpoint_model_type(base_model_id)
                    logger.debug(f"Infered model type is {model_type}")
                else:
                    model_type = "sd1.5"
                    scheduler_config = None
                    sampling_type = "EPS"

                if model_type == "sdxl":
                    if vae_model is None:
                        logger.info("Default VAE: madebyollin/sdxl-vae-fp16-fix")
                    self.pipe = StableDiffusionXLPipeline.from_single_file(
                        base_model_id,
                        vae=AutoencoderKL.from_pretrained(
                            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
                        ),
                        torch_dtype=self.type_model_precision,
                    )
                    class_name = SDXL
                elif model_type == "sd1.5":
                    default_components = {}
                    if not has_baked_vae:
                        sd_vae = "stabilityai/sd-vae-ft-ema"
                        logger.info(
                            "The checkpoint doesn't include a baked VAE, "
                            f"so '{sd_vae}' is being loaded instead."
                        )
                        default_components["vae"] = AutoencoderKL.from_pretrained(
                            sd_vae, torch_dtype=self.type_model_precision
                        )

                    self.pipe = StableDiffusionPipeline.from_single_file(
                        base_model_id,
                        torch_dtype=self.type_model_precision,
                        **default_components,
                    )

                    class_name = SD15
                elif model_type in ["flux-dev", "flux-schnell"]:

                    if "dev" in model_type:
                        repo_flux_conf_transformer = get_flux_components_info()[0]
                    else:
                        repo_flux_conf_transformer = "black-forest-labs/FLUX.1-schnell"

                    transformer = FluxTransformer2DModel.from_single_file(
                        base_model_id,
                        subfolder="transformer",
                        torch_dtype=self.type_model_precision,
                        config=repo_flux_conf_transformer,
                    )

                    if self.env_components is not None:
                        from diffusers import FluxPipeline

                        logger.debug(
                            f"Env components > {self.env_components.keys()}"
                        )
                        self.pipe = FluxPipeline(
                            transformer=transformer,
                            **self.env_components,
                        )
                    else:
                        repo_flux_model = get_flux_components_info()[0]
                        self.pipe = DiffusionPipeline.from_pretrained(
                            repo_flux_model,
                            transformer=transformer,
                            torch_dtype=self.type_model_precision,
                        )

                    class_name = FLUX
                else:
                    raise ValueError(f"Model type {model_type} not supported.")

                if scheduler_config:
                    logger.info(
                        f"Checkpoint schedule prediction type: {sampling_type}"
                    )
                    self.pipe.scheduler.register_to_config(**scheduler_config)

            else:
                try:
                    file_config = hf_hub_download(
                        repo_id=base_model_id,
                        filename="model_index.json",
                    )
                except Exception as e:
                    logger.error(
                        "Unable to obtain the configuration file. Make sure "
                        "you have access to the repository and that it is in"
                        " the diffusers format."
                    )
                    raise e

                # Reading data from the JSON file
                with open(file_config, 'r') as json_config:
                    data_config = json.load(json_config)

                # Searching for the value of the "_class_name" key
                if '_class_name' in data_config:
                    class_name = data_config['_class_name']

                match class_name:

                    case "FluxPipeline":

                        transformer = FluxTransformer2DModel.from_pretrained(
                            base_model_id,
                            subfolder="transformer",
                            torch_dtype=self.type_model_precision,
                        )

                        if self.env_components is not None:
                            from diffusers import FluxPipeline

                            logger.debug(
                                f"Env components > {self.env_components.keys()}"
                            )
                            self.pipe = FluxPipeline(
                                transformer=transformer,
                                **self.env_components,
                            )
                        else:
                            repo_flux_model = get_flux_components_info()[0]

                            self.pipe = DiffusionPipeline.from_pretrained(
                                repo_flux_model,
                                transformer=transformer,
                                torch_dtype=self.type_model_precision,
                            )

                    case "StableDiffusionPipeline":
                        self.pipe = StableDiffusionPipeline.from_pretrained(
                            base_model_id,
                            torch_dtype=self.type_model_precision,
                        )

                    case "StableDiffusionXLPipeline":
                        if vae_model is None:
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
                logger.info(f"VAE: {vae_model}")
                if os.path.isfile(vae_model):
                    self.pipe.vae = AutoencoderKL.from_single_file(
                        vae_model,
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

            # Define base scheduler config
            if self.class_name == FLUX:
                self.pipe.scheduler.register_to_config(
                    shift=(
                        3.0 if self.pipe.transformer.config.guidance_embeds else 1.0
                    ),
                    use_dynamic_shifting=(
                        True if self.pipe.transformer.config.guidance_embeds else False
                    ),
                )

            scheduler_copy = copy.deepcopy(self.pipe.scheduler)

            self.default_scheduler = (
                verify_schedule_integrity(scheduler_copy, base_model_id)
                if self.class_name == SDXL
                else scheduler_copy
            )
            logger.debug(f"Base sampler: {self.default_scheduler}")

            if class_name == SD15:
                self.prompt_embedder = Promt_Embedder_SD1()
            elif class_name == SDXL:
                self.prompt_embedder = Promt_Embedder_SDXL()
            elif class_name == FLUX:
                self.prompt_embedder = Promt_Embedder_FLUX()

        # Load task
        model_id = CONTROLNET_MODEL_IDS[task_name]
        if isinstance(model_id, list):
            if "XL" in class_name:
                model_id = model_id[1]
            else:
                model_id = model_id[0]
        if class_name == FLUX:
            model_id = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro"
        if controlnet_model != "Automatic":
            model_id = controlnet_model

        if task_name in self.model_memory:
            if controlnet_model != "Automatic":
                logger.warning(
                    "The controlnet_model parameter does not support "
                    "retain_task_model_in_cache = True"
                )
            self.pipe = self.model_memory[task_name]
            self.base_model_id = base_model_id
            self.task_name = task_name
            self.vae_model = vae_model
            self.class_name = class_name
            self.pipe.watermark = None
            return

        if (
            (self.task_name != task_name)
            or (self.class_name != class_name)
            or (self.model_id_task != model_id)
        ):
            self.switch_pipe_class(
                class_name,
                task_name,
                model_id,
                enable_pag=False,
                verbose_info=True,
            )

        self.model_id_task = model_id

        self.base_model_id = base_model_id
        self.task_name = task_name
        self.vae_model = vae_model
        self.class_name = class_name

        if self.class_name == SDXL:
            self.pipe.enable_vae_slicing()
            self.pipe.enable_vae_tiling()
            self.pipe.watermark = None

        if retain_task_model_in_cache is True and task_name not in self.model_memory:
            self.model_memory[task_name] = self.pipe

        return

    @cachebox(max_cache_size=2)
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
        tile_blur_sigma: int,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError("No reference image found.")

        if "t2i" in self.task_name:
            preprocessor_name = T2I_PREPROCESSOR_NAME[self.task_name] if t2i_adapter_preprocessor else "None"

        if preprocessor_name in ["None", "None (anime)"] or self.task_name in ["ip2p", "img2img", "pattern", "sdxl_tile_realistic"]:
            return process_basic_task(image, image_resolution)

        params_preprocessor, model_name = get_preprocessor_params(
            image,
            self.task_name,
            preprocessor_name,
            image_resolution,
            preprocess_resolution,
            low_threshold,
            high_threshold,
            value_threshold,
            distance_threshold,
            recolor_gamma_correction,
            tile_blur_sigma,
        )

        if not model_name:
            raise ValueError(
                "Unsupported task name or configuration: "
                f"{self.task_name} - {preprocessor_name}"
            )

        logger.debug(f"Preprocessor: {model_name}")

        self.preprocessor.load(model_name, self.image_preprocessor_cuda_active)
        return self.preprocessor(**params_preprocessor)

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
            if image_mask is None:
                raise ValueError(
                    f"The 'image_mask' parameter is required for the {self.task_name} task."
                )
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

        init_image = process_basic_task(image, image_resolution)
        control_mask = process_basic_task(array_rgb_mask, image_resolution)
        control_image = make_inpaint_condition(init_image, control_mask)

        return init_image, control_mask, control_image

    def get_scheduler(self, name):
        if "Flow" in name and self.class_name != FLUX:
            name = name.replace("FlowMatch ", "")
        elif "Flow" not in name and self.class_name == FLUX:
            name = "FlowMatch DPM++ 2M"

        if name in SCHEDULER_CONFIG_MAP:
            scheduler_class, config = SCHEDULER_CONFIG_MAP[name]
            # return scheduler_class.from_config(self.pipe.scheduler.config, **config)
            # beta self.default_scheduler
            return scheduler_class.from_config(self.default_scheduler.config, **config)
        else:
            raise ValueError(f"Scheduler with name {name} not found. Valid schedulers: {', '.join(scheduler_names)}")

    @cachebox(max_cache_size=2)
    def create_prompt_embeds(
        self,
        prompt,
        negative_prompt,
        textual_inversion,
        clip_skip,
        syntax_weights,
    ):
        if self.embed_loaded != textual_inversion and textual_inversion != []:
            self.prompt_embedder.apply_ti(
                self.class_name,
                textual_inversion,
                self.pipe,
                self.device,
                self.gui_active
            )
            self.embed_loaded = textual_inversion

        # Prompt weights for textual inversion
        if hasattr(self.pipe, "maybe_convert_prompt"):
            prompt_ti = self.pipe.maybe_convert_prompt(prompt, self.pipe.tokenizer)
            negative_prompt_ti = self.pipe.maybe_convert_prompt(
                negative_prompt, self.pipe.tokenizer
            )
        else:
            prompt_ti = prompt
            negative_prompt_ti = negative_prompt

        # separate the multi-vector textual inversion by comma
        if self.embed_loaded != []:
            prompt_ti = add_comma_after_pattern_ti(prompt_ti)
            negative_prompt_ti = add_comma_after_pattern_ti(negative_prompt_ti)

        prompt_emb, negative_prompt_emb, compel = self.prompt_embedder(
            prompt_ti,
            negative_prompt_ti,
            syntax_weights,
            self.pipe,
            clip_skip,
            self.compel if hasattr(self, "compel") else None,
        )

        return prompt_emb, negative_prompt_emb

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
                    self.last_lora_error = str(e)
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
                vit_repo[0],
                subfolder=vit_repo[1],
                use_safetensors=True,
                torch_dtype=self.type_model_precision,
            ).to(self.device, dtype=self.type_model_precision)
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

                    release_resources()

                    self.pipe.unet.encoder_hid_proj.image_projection_layers[i].clip_embeds = clip_embeds.to(dtype=self.type_model_precision)
                    if "plusv2" in ip_weight:
                        self.pipe.unet.encoder_hid_proj.image_projection_layers[i].shortcut = True
                    else:
                        self.pipe.unet.encoder_hid_proj.image_projection_layers[i].shortcut = False

                release_resources()

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

    def load_lora_on_the_fly(
        self,
        lora_A=None, lora_scale_A=1.0,
        lora_B=None, lora_scale_B=1.0,
        lora_C=None, lora_scale_C=1.0,
        lora_D=None, lora_scale_D=1.0,
        lora_E=None, lora_scale_E=1.0,
        lora_F=None, lora_scale_F=1.0,
        lora_G=None, lora_scale_G=1.0,
    ):
        """
        Dynamically applies LoRA weights without fusing them.
        LoRAs load quickly, but they need more RAM and can make the inference process take longer.

        Note:
            Ensure any merged LoRA with `model.lora_merge()` remains loaded before using this method.
        """

        current_lora_list = [
            lora_A,
            lora_B,
            lora_C,
            lora_D,
            lora_E,
            lora_F,
            lora_G
        ]

        current_lora_scale_list = [
            lora_scale_A,
            lora_scale_B,
            lora_scale_C,
            lora_scale_D,
            lora_scale_E,
            lora_scale_F,
            lora_scale_G
        ]

        lora_status = [None] * self.num_loras

        if self.lora_memory == current_lora_list and self.lora_scale_memory == current_lora_scale_list:
            for single_lora in self.lora_memory:
                if single_lora is not None:
                    logger.info(f"LoRA in memory: {single_lora}")
            pass
        elif (
            self.lora_memory == [None] * self.num_loras
            and current_lora_list == [None] * self.num_loras
        ):
            pass
        else:
            self.create_prompt_embeds.memory.clear()

            lora_status = load_no_fused_lora(
                self.pipe, self.num_loras,
                current_lora_list, current_lora_scale_list
            )

        logger.debug(str(self.pipe.get_active_adapters()))

        self.lora_memory = current_lora_list
        self.lora_scale_memory = current_lora_scale_list

        return lora_status

    def lora_merge(
        self,
        lora_A=None, lora_scale_A=1.0,
        lora_B=None, lora_scale_B=1.0,
        lora_C=None, lora_scale_C=1.0,
        lora_D=None, lora_scale_D=1.0,
        lora_E=None, lora_scale_E=1.0,
        lora_F=None, lora_scale_F=1.0,
        lora_G=None, lora_scale_G=1.0,
    ):
        current_lora_list = [
            lora_A,
            lora_B,
            lora_C,
            lora_D,
            lora_E,
            lora_F,
            lora_G
        ]

        current_lora_scale_list = [
            lora_scale_A,
            lora_scale_B,
            lora_scale_C,
            lora_scale_D,
            lora_scale_E,
            lora_scale_F,
            lora_scale_G
        ]

        lora_status = [None] * self.num_loras

        if self.lora_memory == current_lora_list and self.lora_scale_memory == current_lora_scale_list:
            for single_lora in self.lora_memory:
                if single_lora is not None:
                    logger.info(f"LoRA in memory: {single_lora}")
            pass
        elif (
            self.lora_memory == [None] * self.num_loras
            and current_lora_list == [None] * self.num_loras
        ):
            pass
        else:
            self.create_prompt_embeds.memory.clear()

            logger.debug("_un, re and load_ lora")

            for l_memory, scale_memory in zip(self.lora_memory, self.lora_scale_memory):
                self.process_lora(l_memory, scale_memory, unload=True)

            for i, (l_new, scale_new) in enumerate(zip(current_lora_list, current_lora_scale_list)):
                lora_status[i] = self.process_lora(l_new, scale_new)

        self.lora_memory = current_lora_list
        self.lora_scale_memory = current_lora_scale_list

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
        schedule_type: str = "Automatic",
        schedule_prediction_type: str = "Automatic",
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
        lora_F: Optional[str] = None,
        lora_scale_F: float = 1.0,
        lora_G: Optional[str] = None,
        lora_scale_G: float = 1.0,
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
        tile_blur_sigma: int = 9,
        controlnet_conditioning_scale: float = 1.0,
        control_guidance_start: float = 0.0,
        control_guidance_end: float = 1.0,
        t2i_adapter_preprocessor: bool = True,
        t2i_adapter_conditioning_scale: float = 1.0,
        t2i_adapter_conditioning_factor: float = 1.0,

        upscaler_model_path: Optional[str] = None,
        upscaler_increases_size: float = 1.5,
        upscaler_tile_size: int = 192,  # max 512, step 16
        upscaler_tile_overlap: int = 8,  # max 48
        hires_steps: int = 25,
        hires_denoising_strength: float = 0.35,
        hires_prompt: str = "",
        hires_negative_prompt: str = "",
        hires_sampler: str = "Use same sampler",
        hires_schedule_type: str = "Use same schedule type",
        hires_guidance_scale: float = -1.,

        face_restoration_model: Optional[str] = None,
        face_restoration_visibility: float = 1.0,
        face_restoration_weight: float = 0.5,

        ip_adapter_image: Optional[Any] = [],  # str Image
        ip_adapter_mask: Optional[Any] = [],  # str Image
        ip_adapter_model: Optional[Any] = [],  # str
        ip_adapter_scale: Optional[Any] = [1.0],  # float
        ip_adapter_mode: Optional[Any] = ["original"],  # str: original, style, layout, style+layout

        loop_generation: int = 1,
        display_images: bool = False,
        image_display_scale: int = 1.,
        save_generated_images: bool = True,
        filename_pattern: str = "model,seed",
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
        **kwargs,
    ):

        """
        The call function for the generation.

        Args:
            prompt (str , optional):
                The prompt to guide image generation.
            negative_prompt (str , optional):
                The prompt to guide what to not include in image generation. Ignored when not using guidance (`guidance_scale < 1`).
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
                The sampler algorithm that defines how noise is gradually removed.
                To see all the valid sampler names, use the following code:

                ```python
                from stablepy import scheduler_names
                print(scheduler_names)
                ```
            schedule_type (str, optional, defaults to "Automatic"):
                The pattern controlling the rate at which noise is removed across each generation step.
                To see all the valid schedule_type names, use the following code:

                ```python
                from stablepy import SCHEDULE_TYPE_OPTIONS
                print(SCHEDULE_TYPE_OPTIONS)
                ```
            schedule_prediction_type (str, optional, defaults to "Automatic"):
                Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
                `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
                Video](https://imagen.research.google/video/paper.pdf) paper).
                To see all the valid schedule_prediction_type names, use the following code:

                ```python
                from stablepy import SCHEDULE_PREDICTION_TYPE_OPTIONS
                print(SCHEDULE_PREDICTION_TYPE_OPTIONS)
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
            lora_F (str, optional):
                Placeholder for lora E parameter.
            lora_scale_F (float, optional, defaults to 1.0):
                Placeholder for lora scale E parameter.
            lora_G (str, optional):
                Placeholder for lora E parameter.
            lora_scale_G (float, optional, defaults to 1.0):
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
                This is the path of the model that will be used for the upscale; on the other hand,
                you can also simply use any of the built-in upscalers like 'Nearest', 'Latent', 'SwinIR 4x', etc.
                that can be consulted in the following code:

                ```python
                from stablepy import ALL_BUILTIN_UPSCALERS
                print(ALL_BUILTIN_UPSCALERS)
                ```
            upscaler_increases_size (float, optional, defaults to 1.5):
                Placeholder for upscaler increases size parameter.
            upscaler_tile_size (int, optional, defaults to 192):
                Tile if use a upscaler model.
            upscaler_tile_overlap (int, optional, defaults to 8):
                Tile overlap if use a upscaler model.
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
            hires_schedule_type (str, optional, defaults to "Use same schedule type"):
                The schedule type used for the hires generation process. If not specified, the main schedule will be used.
            hires_guidance_scale (float, optional, defaults to -1.):
                The guidance scale used for the hires generation process.
                If the value is set to -1. the main guidance_scale will be used.
            face_restoration_model (str, optional default None):
                This is the name of the face restoration model that will be used.

                To see all the valid face restoration model names, use the following code:

                ```python
                from stablepy import FACE_RESTORATION_MODELS
                print(FACE_RESTORATION_MODELS)
                ```
            face_restoration_visibility (float, optional, defaults to 1.):
                The visibility of the restored face's changes.
            face_restoration_weight (float, optional, defaults to 1.):
                The weight value used for the CodeFormer model.
            image (Any, optional):
                The image to be used for the Inpaint, ControlNet, or T2I adapter.
            preprocessor_name (str, optional, defaults to "None"):
                Preprocessor name for ControlNet tasks.
                To see the mapping of tasks with their corresponding preprocessor names, use the following code:
                ```python
                from stablepy import TASK_AND_PREPROCESSORS
                print(TASK_AND_PREPROCESSORS)
                ```
            preprocess_resolution (int, optional, defaults to 512):
                Preprocess resolution for the Inpaint, ControlNet, or T2I adapter.
                This is the resolution used by the preprocessor to preprocess the image.
                Lower resolution can work faster but provides fewer details,
                while higher resolution gives more detail but can be slower and requires more resources.
            image_resolution (int, optional, defaults to 512):
                Image resolution for the Img2Img, Inpaint, ControlNet, or T2I adapter.
                This is the maximum resolution the image will be scaled to while maintaining its proportions,
                based on the original dimensions.
                For example, if you have a 512x512 image and set image_resolution=1024,
                it will be resized to 1024x1024 during image generation.
            image_mask (Any, optional):
                Path image mask for the Inpaint and Repaint tasks.
            strength (float, optional, defaults to 0.35):
                Strength parameter for the Inpaint, Repaint and Img2Img.
            low_threshold (int, optional, defaults to 100):
                Low threshold parameter for ControlNet and T2I Adapter Canny.
            high_threshold (int, optional, defaults to 200):
                High threshold parameter for ControlNet and T2I Adapter Canny.
            value_threshold (float, optional, defaults to 0.1):
                Value threshold parameter for ControlNet MLSD.
            distance_threshold (float, optional, defaults to 0.1):
                Distance threshold parameter for ControlNet MLSD.
            recolor_gamma_correction (float, optional, defaults to 1.0):
                Gamma correction parameter for ControlNet Recolor.
            tile_blur_sigma (int, optional, defaults to 9.0):
                Blur sigma paramater for ControlNet Tile.
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
            image_display_scale (float, optional, defaults to 1.):
                The proportional scale of the displayed image in the notebook.
            save_generated_images (bool, optional, defaults to True):
                By default, the generated images are saved in the current location within the 'images' folder. You can disable this with this parameter.
            filename_pattern (str , optional, defaults to "model,seed"):
                Sets the name that will be used to save the images.
                This name can be any text or a specific key, and each value needs to be separated by a comma.
                You can check the list of valid keys:
                ```python
                from stablepy import VALID_FILENAME_PATTERNS
                print(VALID_FILENAME_PATTERNS)
                ```
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
                Improves generation time; currently disabled due to quality issues with LoRA.
            gui_active (bool, optional, defaults to False):
                utility when used with a GUI, it changes the behavior especially by displaying confirmation messages or options.
            **kwargs (dict, optional):
                kwargs is used to pass additional parameters to the Diffusers pipeline. This allows for flexibility
                when specifying optional settings like guidance_rescale, eta, cross_attention_kwargs, and more.


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
        if self.task_name == "":
            raise RuntimeError(
                "Some components of the model did not "
                "load correctly. Please reload the model"
            )
        if self.task_name != "txt2img" and image is None:
            raise ValueError(
                f"The 'image' parameter is required for the {self.task_name} task."
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

        sampler, schedule_type, msg_schedule = check_scheduler_compatibility(
            self.class_name,
            sampler,
            schedule_type,
        )
        if msg_schedule:
            logger.warning(msg_schedule)

        face_restoration_device = self.device.type
        self.gui_active = gui_active

        pp = CURRENT_TASK_PARAMS(
            disable_progress_bar=disable_progress_bar,
            display_images=display_images,
            image_display_scale=image_display_scale,
            image_storage_location=image_storage_location,
            save_generated_images=save_generated_images,
            retain_compel_previous_load=retain_compel_previous_load,
            hires_steps=hires_steps,
            hires_before_adetailer=hires_before_adetailer,
            hires_after_adetailer=hires_after_adetailer,
            upscaler_model_path=upscaler_model_path,
            upscaler_increases_size=upscaler_increases_size,
            upscaler_tile_size=upscaler_tile_size,
            upscaler_tile_overlap=upscaler_tile_overlap,
            face_restoration_model=face_restoration_model,
            face_restoration_visibility=face_restoration_visibility,
            face_restoration_weight=face_restoration_weight,
            face_restoration_device=face_restoration_device,
            loop_generation=loop_generation,
            generator_in_cpu=generator_in_cpu,
            num_images=num_images,
            seed=seed,
            adetailer_A=adetailer_A,
            adetailer_B=adetailer_B,
        )

        if self.pipe is None:
            self.load_pipe(
                self.base_model_id,
                task_name=self.task_name,
                vae_model=self.vae_model,
                type_model_precision=self.type_model_precision,
                retain_task_model_in_cache=False,
                reload=True,
                controlnet_model=self.model_id_task,
            )
        elif (
            self.class_name == FLUX
            and (self.pipe.text_encoder is None or self.pipe.transformer is None)
        ):
            logger.info("Reloading flux pipeline")
            original_device = self.device
            self.device = torch.device("cpu")
            self.load_pipe(
                self.base_model_id,
                task_name=self.task_name,
                vae_model=self.vae_model,
                type_model_precision=self.type_model_precision,
                retain_task_model_in_cache=False,
                reload=True,
                controlnet_model=self.model_id_task,
            )
            self.device = original_device

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
        self.pipe.set_progress_bar_config(disable=pp.disable_progress_bar)

        xformers_memory_efficient_attention = False  # disabled
        if xformers_memory_efficient_attention and torch.cuda.is_available():
            self.pipe.disable_xformers_memory_efficient_attention()

        if self.class_name != FLUX:
            self.pipe.to(self.device)
        else:
            self.pipe.text_encoder.to(self.device)
            self.pipe.text_encoder_2.to(self.device)
            logger.debug(str(self.pipe._execution_device))  # alternative patch to secuential_cpu_offload()

            origin_device = self.device

            def _execution_device(self):
                logger.debug(f"patch device is {origin_device}")
                return origin_device

            self.pipe.__class__._execution_device = property(_execution_device)
            logger.debug(str(self.pipe._execution_device))

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
            lora_F, lora_scale_F,
            lora_G, lora_scale_G,
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

        if ip_adapter_image:
            if self.class_name == FLUX:
                raise ValueError("IP adapter is not currently supported with Flux")
            for ipa_ml in ip_adapter_model:
                if ipa_ml not in IP_ADAPTER_MODELS[self.class_name]:
                    raise ValueError(
                        f"Invalid IP adapter model '{ipa_ml}' for {self.class_name}. "
                        f"Valid models are: {', '.join(IP_ADAPTER_MODELS[self.class_name].keys())}"
                    )

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
        if FreeU and self.class_name != FLUX:
            logger.info("FreeU active")
            if self.class_name == SD15:
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
        if hasattr(self, "compel") and not pp.retain_compel_previous_load:
            del self.compel

        prompt_emb, negative_prompt_emb = self.create_prompt_embeds(
            prompt=prompt,
            negative_prompt=negative_prompt,
            textual_inversion=textual_inversion,
            clip_skip=clip_skip,
            syntax_weights=syntax_weights,
        )

        if self.class_name == SDXL:
            # Additional prompt for SDXL
            conditioning, pooled = prompt_emb.clone(), negative_prompt_emb.clone()
            prompt_emb = negative_prompt_emb = None
        elif self.class_name == FLUX:
            conditioning, pooled = prompt_emb.clone(), negative_prompt_emb.clone()
            prompt_emb = negative_prompt_emb = None

        try:
            self.pipe.scheduler = self.get_scheduler(sampler)
            configure_scheduler(
                self.pipe, schedule_type, schedule_prediction_type
            )
        except Exception as e:
            logger.debug(f"{e}")
            release_resources()
            raise RuntimeError("Error in sampler, please try again")

        self.pipe.safety_checker = None

        # Reference image
        if self.task_name != "txt2img":
            array_rgb = convert_image_to_numpy_array(image, gui_active)

        control_image = None

        # Run preprocess
        if self.task_name in ["inpaint", "repaint"]:
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
                tile_blur_sigma=tile_blur_sigma,
            )

        # Task Parameters
        pipe_params_config = {
            "prompt": None,
            "negative_prompt": None,
            "num_inference_steps": num_steps,
            "guidance_scale": guidance_scale,
            "clip_skip": None,
            "num_images_per_prompt": pp.num_images,
        }

        if hasattr(self.pipe, "set_pag_applied_layers"):
            pipe_params_config["pag_scale"] = float(pag_scale)

        pipe_params_config.update(
            ays_timesteps(self.class_name, schedule_type, num_steps)
        )

        if self.task_name == "txt2img":
            pipe_params_config["height"] = img_height
            pipe_params_config["width"] = img_width
        else:
            pipe_params_config["image"] = control_image
            logger.info(f"Image resolution: {str(control_image.size)}")

        if self.class_name == SD15:
            pipe_params_config["prompt_embeds"] = prompt_emb
            pipe_params_config["negative_prompt_embeds"] = negative_prompt_emb

            if self.task_name in ["inpaint", "repaint"]:
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

        elif self.class_name == SDXL:
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

                if "t2i" in self.task_name:
                    pipe_params_config["adapter_conditioning_scale"] = float(t2i_adapter_conditioning_scale)
                    pipe_params_config["adapter_conditioning_factor"] = float(t2i_adapter_conditioning_factor)
                else:
                    pipe_params_config[
                        "controlnet_conditioning_scale"
                    ] = float(controlnet_conditioning_scale)
                    pipe_params_config["control_guidance_start"] = float(control_guidance_start)
                    pipe_params_config["control_guidance_end"] = float(control_guidance_end)

                    if self.task_name == "repaint":
                        pipe_params_config["strength"] = strength
                        pipe_params_config["mask_image"] = control_mask
                        pipe_params_config["height"] = control_image.size[1]
                        pipe_params_config["width"] = control_image.size[0]

                    if "XLControlNetUnion" in self.pipe.__class__.__name__:
                        pipe_params_config["control_mode"] = SDXL_CN_UNION_PROMAX_MODES[self.task_name]

                    if self.pipe.__class__.__name__ == "StableDiffusionXLControlNetUnionPipeline":
                        pipe_params_config["control_image"] = pipe_params_config.pop('image')
                    elif self.pipe.__class__.__name__ == "StableDiffusionXLControlNetUnionInpaintPipeline":
                        def get_union_cn_img(image_cn, mask_cn):
                            controlnet_img = image_cn.copy()
                            controlnet_img_np = np.array(controlnet_img)
                            mask_np = np.array(mask_cn)
                            controlnet_img_np[mask_np > 0] = 0
                            return Image.fromarray(controlnet_img_np)
                        pipe_params_config["control_image"] = get_union_cn_img(control_image, control_mask)
                    elif self.pipe.__class__.__name__ == "StableDiffusionXLControlNetInpaintPipeline":
                        # self.preprocessor.load("Canny", self.image_preprocessor_cuda_active)
                        # pipe_params_config["control_image"] = self.preprocessor(control_image)
                        pipe_params_config["control_image"] = control_image

            elif self.task_name == "img2img":
                pipe_params_config["strength"] = strength

        elif self.class_name == FLUX:
            pipe_params_config.pop("negative_prompt", None)
            pipe_params_config.pop("clip_skip", None)
            pipe_params_config["prompt_embeds"] = conditioning
            pipe_params_config["pooled_prompt_embeds"] = pooled

            if self.task_name != "txt2img":
                pipe_params_config["height"] = img_height = control_image.size[1]
                pipe_params_config["width"] = img_width = control_image.size[0]

            if self.task_name == "inpaint":
                pipe_params_config["strength"] = strength
                pipe_params_config["mask_image"] = control_mask
            elif self.task_name not in ["txt2img", "inpaint", "img2img"]:
                pipe_params_config["control_image"] = pipe_params_config.pop("image")

                cn_mode = FLUX_CN_UNION_MODES[self.task_name]
                if isinstance(cn_mode, list):
                    if preprocessor_name is None or "None" in preprocessor_name:
                        cn_mode = random.choice(cn_mode)
                    else:
                        cn_mode = cn_mode[1]
                pipe_params_config["control_mode"] = cn_mode

                pipe_params_config[
                    "controlnet_conditioning_scale"
                ] = float(controlnet_conditioning_scale)
                # pipe_params_config["control_guidance_start"] = float(control_guidance_start)
                # pipe_params_config["control_guidance_end"] = float(control_guidance_end)
            elif self.task_name == "img2img":
                pipe_params_config["strength"] = strength

        if self.ip_adapter_config:
            # maybe need cache embeds
            ip_adapter_embeds, ip_adapter_masks = self.get_ip_embeds(
                guidance_scale, ip_images, pp.num_images, ip_masks
            )

            pipe_params_config["ip_adapter_image_embeds"] = ip_adapter_embeds
            if ip_adapter_masks:
                pipe_params_config["cross_attention_kwargs"] = {
                    "ip_adapter_masks": ip_adapter_masks
                }

        # detailfix params and pipe global
        if pp.adetailer_A or pp.adetailer_B:

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
                if adetailer_A_params.get("inpaint_only", False) is True or adetailer_B_params.get("inpaint_only", False) is True:
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
                configure_scheduler(
                    detailfix_pipe, "Automatic", schedule_prediction_type
                )
            adetailer_A_params.pop("sampler", None)
            if adetailer_B_params.get("sampler", "Use same sampler") != "Use same sampler":
                logger.debug("detailfix_pipe will use the sampler from adetailer_B")
                detailfix_pipe.scheduler = self.get_scheduler(adetailer_A_params["sampler"])
                configure_scheduler(
                    detailfix_pipe, "Automatic", schedule_prediction_type
                )
            adetailer_B_params.pop("sampler", None)

            detailfix_pipe.set_progress_bar_config(leave=leave_progress_bar)
            detailfix_pipe.set_progress_bar_config(disable=pp.disable_progress_bar)
            if self.class_name != FLUX:
                detailfix_pipe.to(self.device)
            else:
                detailfix_pipe.__class__._execution_device = property(_execution_device)
            release_resources()
        else:
            detailfix_pipe = None

        if pp.adetailer_A:
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
            if self.class_name == SD15:
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

            elif self.class_name == SDXL:
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

            elif self.class_name == FLUX:
                # Flux detailfix
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
                detailfix_params_A["prompt_embeds"] = conditioning_detailfix_A
                detailfix_params_A["pooled_prompt_embeds"] = pooled_detailfix_A

            logger.debug(f"detailfix A prompt empty {prompt_empty_detailfix_A, negative_prompt_empty_detailfix_A}")
            if not prompt_empty_detailfix_A or not negative_prompt_empty_detailfix_A:
                logger.debug(f"Prompts detailfix A {prompt_df_A, negative_prompt_df_A}")
            logger.debug(f"Pipe params detailfix A \n{detailfix_params_A}")
            logger.debug(f"Params detailfix A \n{adetailer_A_params}")

            pp.set_params(detailfix_params_A=detailfix_params_A)

        if pp.adetailer_B:
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
            if self.class_name == SD15:
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

            elif self.class_name == SDXL:
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

            elif self.class_name == FLUX:
                # Flux detailfix
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
                detailfix_params_B.pop("prompt", None)
                detailfix_params_B.pop("negative_prompt", None)
                detailfix_params_B["prompt_embeds"] = conditioning_detailfix_B
                detailfix_params_B["pooled_prompt_embeds"] = pooled_detailfix_B

            logger.debug(f"detailfix B prompt empty {prompt_empty_detailfix_B, negative_prompt_empty_detailfix_B}")
            if not prompt_empty_detailfix_B or not negative_prompt_empty_detailfix_B:
                logger.debug(f"Prompts detailfix B {prompt_df_B, negative_prompt_df_B}")
            logger.debug(f"Pipe params detailfix B \n{detailfix_params_B}")
            logger.debug(f"Params detailfix B \n{adetailer_B_params}")

            pp.set_params(detailfix_params_B=detailfix_params_B)

        if pp.hires_steps > 1 and pp.upscaler_model_path is not None:
            # Hires params BASE
            hires_params_config = {
                "prompt": None,
                "negative_prompt": None,
                "num_inference_steps": pp.hires_steps,
                "guidance_scale": hires_guidance_scale if hires_guidance_scale >= 0 else guidance_scale,
                "clip_skip": None,
                "strength": hires_denoising_strength,
            }
            if self.class_name == SD15:
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
            if self.class_name == SD15:
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
            elif self.class_name == SDXL:
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

            elif self.class_name == FLUX:
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

                hires_params_config.pop("clip_skip", None)
                hires_params_config.pop('prompt', None)
                hires_params_config.pop('negative_prompt', None)
                hires_params_config["prompt_embeds"] = hires_conditioning
                hires_params_config["pooled_prompt_embeds"] = hires_pooled

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
            hires_sampler_fix, hires_sch_type_fix, msg_hires_fix = check_scheduler_compatibility(
                self.class_name,
                hires_sampler if hires_sampler != "Use same sampler" else sampler,
                hires_schedule_type if hires_schedule_type != "Use same schedule type" else schedule_type,
            )
            if msg_hires_fix:
                logger.warning(f"Hires > {msg_hires_fix}")

            if sampler != hires_sampler_fix or schedule_type != hires_sch_type_fix:
                logger.debug("New hires sampler")
                hires_pipe.scheduler = self.get_scheduler(hires_sampler_fix)
                configure_scheduler(
                    hires_pipe, hires_sch_type_fix, schedule_prediction_type
                )
            hires_params_config.update(
                ays_timesteps(self.class_name, hires_sch_type_fix, pp.hires_steps)
            )

            hires_pipe.set_progress_bar_config(leave=leave_progress_bar)
            hires_pipe.set_progress_bar_config(disable=pp.disable_progress_bar)
            if self.class_name != FLUX:
                hires_pipe.to(self.device)
            else:
                hires_pipe.__class__._execution_device = property(_execution_device)
            release_resources()

            if (
                pp.upscaler_model_path in LATENT_UPSCALERS
                and ((not pp.adetailer_A and not pp.adetailer_B and not pp.face_restoration_model) or pp.hires_before_adetailer)
            ):
                pipe_params_config["output_type"] = "latent"

        else:
            hires_params_config = {}
            hires_pipe = None

        # Debug info
        try:
            logger.debug(f"INFO PIPE: {self.pipe.__class__.__name__}")
            logger.debug(f"text_encoder_type: {self.pipe.text_encoder.dtype}")
            if hasattr(self.pipe, "unet"):
                logger.debug(f"unet_type: {self.pipe.unet.dtype}")
            else:
                logger.debug(f"transformer_type: {self.pipe.transformer.dtype}")
                logger.debug(f"transformer_config: {self.pipe.transformer.config}")
            logger.debug(f"vae_type: {self.pipe.vae.dtype}")
            logger.debug(f"pipe_type: {self.pipe.dtype}")
            logger.debug(f"scheduler_main_pipe: {self.pipe.scheduler}")
            if pp.adetailer_A or pp.adetailer_B:
                logger.debug(f"scheduler_detailfix: {detailfix_pipe.scheduler}")
            if pp.hires_steps > 1 and pp.upscaler_model_path is not None:
                logger.debug(f"scheduler_hires: {hires_pipe.scheduler}")
        except Exception as e:
            logger.debug(f"{str(e)}")

        extra_metadata = extra_string_metadata(
            [
                self.vae_model,
                pag_scale if hasattr(self.pipe, "set_pag_applied_layers") else 0,
                self.FreeU,
                pp.upscaler_model_path,
                pp.upscaler_increases_size,
                pp.hires_steps,
                hires_denoising_strength,
                self.lora_memory,
                self.lora_scale_memory,
            ]
        )

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
            schedule_type,
            extra_metadata,
        ]

        filename_pattern = assemble_filename_pattern(
            filename_pattern, metadata
        )

        pp.set_params(
            metadata=metadata, filename_pattern=filename_pattern,
            adetailer_A_params=adetailer_A_params, adetailer_B_params=adetailer_B_params,
            hires_params_config=hires_params_config,
        )

        # === RUN PIPE === #
        handle_task = self.start_work if not image_previews else self.start_stream

        return handle_task(
            control_image,
            pipe_params_config,
            detailfix_pipe,
            hires_pipe,
            pp,
            kwargs,
        )

    def post_processing(
        self,
        control_image,
        detailfix_pipe,
        hires_pipe,
        seeds,
        generators,
        images,
        pp,
    ):

        if self.class_name == FLUX:
            if pp.upscaler_model_path is None and not pp.adetailer_A and not pp.adetailer_B and pp.loop_generation == 1:
                if os.getenv("SPACES_ZERO_GPU"):
                    self.pipe.transformer = None
                else:
                    self.pipe.transformer.to("cpu")
            if hasattr(self.pipe, "controlnet") and pp.loop_generation == 1:
                if os.getenv("SPACES_ZERO_GPU"):
                    self.pipe.controlnet = None
                else:
                    self.pipe.controlnet.to("cpu")

            if pp.upscaler_model_path not in LATENT_UPSCALERS:
                self.pipe.vae.to(self.device)
                release_resources()
                images.to(self.device)

                with torch.no_grad():
                    latents = self.pipe._unpack_latents(images, pp.metadata[9], pp.metadata[8], self.pipe.vae_scale_factor)
                    latents = (latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
                    image = self.pipe.vae.decode(latents, return_dict=False)[0]
                    images = self.pipe.image_processor.postprocess(image, output_type="pil")
                    if not os.getenv("SPACES_ZERO_GPU"):
                        self.pipe.vae.to("cpu")

        if isinstance(images, torch.Tensor):
            images = [tl.unsqueeze(0) for tl in torch.unbind(images, dim=0)]

        release_resources()

        if pp.hires_before_adetailer and pp.upscaler_model_path is not None:
            logger.debug(
                "Hires before; same seed for each image (no batch)"
            )
            images = process_images_high_resolution(
                images,
                pp.upscaler_model_path,
                pp.upscaler_increases_size,
                pp.upscaler_tile_size, pp.upscaler_tile_overlap,
                pp.hires_steps, pp.hires_params_config,
                self.task_name,
                generators[0] if isinstance(generators, list) else generators,  # pipe_params_config["generator"][0], # no generator
                hires_pipe,
                pp.disable_progress_bar,
            )

        if pp.face_restoration_model:
            images = batch_process_face_restoration(
                images,
                pp.face_restoration_model,
                pp.face_restoration_visibility,
                pp.face_restoration_weight,
                pp.face_restoration_device,
            )

            # Adetailer stuff
        if pp.adetailer_A or pp.adetailer_B:
            # image_pil_list = []
            # for img_single in images:
            # image_ad = img_single.convert("RGB")
            # image_pil_list.append(image_ad)

            if pp.adetailer_A:
                images = ad_model_process(
                    pipe_params_df=pp.detailfix_params_A,
                    detailfix_pipe=detailfix_pipe,
                    image_list_task=images,
                    **pp.adetailer_A_params,
                )
            if pp.adetailer_B:
                images = ad_model_process(
                    pipe_params_df=pp.detailfix_params_B,
                    detailfix_pipe=detailfix_pipe,
                    image_list_task=images,
                    **pp.adetailer_B_params,
                )

                # del detailfix_pipe
            release_resources()

        if pp.hires_after_adetailer and pp.upscaler_model_path is not None:
            logger.debug(
                "Hires after; same seed for each image (no batch)"
            )
            images = process_images_high_resolution(
                images,
                pp.upscaler_model_path,
                pp.upscaler_increases_size,
                pp.upscaler_tile_size, pp.upscaler_tile_overlap,
                pp.hires_steps, pp.hires_params_config,
                self.task_name,
                generators[0] if isinstance(generators, list) else generators,  # pipe_params_config["generator"][0], # no generator
                hires_pipe,
                pp.disable_progress_bar,
            )

        if self.task_name not in ["txt2img", "inpaint", "img2img", "repaint"]:
            images = [control_image] + images
            valid_seeds = [0] + seeds
        else:
            valid_seeds = seeds

        logger.info(f"Seeds: {valid_seeds}")

        # Show images
        if pp.display_images:
            if pp.image_display_scale != 1.0:
                resized_images = []
                for img in images:
                    img_copy = img.copy()
                    img_copy.thumbnail(
                        (int(img_copy.width * pp.image_display_scale), int(img_copy.height * pp.image_display_scale)),
                        Image.LANCZOS
                    )
                    resized_images.append(img_copy)
                mediapy.show_images(resized_images)
            else:
                mediapy.show_images(images)

            if pp.loop_generation > 1:
                time.sleep(0.5)

        # List images and save
        image_list = []
        image_metadata = []

        for image_, seed_ in zip(images, valid_seeds):

            pp.metadata[7] = seed_
            image_generation_data = get_string_metadata(pp.metadata)

            image_path = "not saved in storage"
            if pp.save_generated_images:
                sfx = pp.filename_pattern
                if "_STABLEPYSEEDKEY_" in sfx:
                    sfx = sfx.replace("_STABLEPYSEEDKEY_", str(seed_))
                image_path = save_pil_image_with_metadata(
                    image_, pp.image_storage_location, image_generation_data, sfx
                )

            image_list.append(image_path)
            image_metadata.append(image_generation_data)

        release_resources()

        if image_list[0] != "not saved in storage":
            logger.info(image_list)

        return images, image_list, image_metadata

    def start_work(
        self,
        control_image,
        pipe_params_config,
        detailfix_pipe,
        hires_pipe,
        pp,
        kwargs,
    ):
        for i in range(pp.loop_generation):
            # number seed
            seeds, generators = get_seeds(pp, self.task_name, self.device)
            pipe_params_config["generator"] = generators

            if self.class_name == FLUX:
                if os.getenv("SPACES_ZERO_GPU"):
                    self.pipe.text_encoder = None
                    self.pipe.text_encoder_2 = None
                else:
                    self.pipe.text_encoder.to("cpu")
                    self.pipe.text_encoder_2.to("cpu")
                release_resources()
                if self.task_name == "txt2img":
                    self.pipe.transformer.to(self.device)
                release_resources()
                pipe_params_config["output_type"] = "latent"
                # self.pipe.__class__._execution_device = property(_execution_device)

            validate_and_update_params(self.pipe.__class__, kwargs, pipe_params_config)

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
                control_image,
                detailfix_pipe,
                hires_pipe,
                seeds,
                generators,
                images,
                pp,
            )

        if hasattr(self, "compel") and not pp.retain_compel_previous_load:
            del self.compel
        release_resources()
        return images, [seeds, image_list, image_metadata]

    def start_stream(
        self,
        control_image,
        pipe_params_config,
        detailfix_pipe,
        hires_pipe,
        pp,
        kwargs,
    ):
        for i in range(pp.loop_generation):
            # number seed
            seeds, generators = get_seeds(pp, self.task_name, self.device)
            pipe_params_config["generator"] = generators

            if self.class_name == FLUX:
                if os.getenv("SPACES_ZERO_GPU"):
                    self.pipe.text_encoder = None
                    self.pipe.text_encoder_2 = None
                else:
                    self.pipe.text_encoder.to("cpu")
                    self.pipe.text_encoder_2.to("cpu")
                release_resources()
                if self.task_name == "txt2img":
                    self.pipe.transformer.to(self.device)
                release_resources()
                pipe_params_config["output_type"] = "latent"
                # self.pipe.__class__._execution_device = property(_execution_device)
                self.metadata = pp.metadata

            validate_and_update_params(self.pipe.__class__, kwargs, pipe_params_config)

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
                control_image,
                detailfix_pipe,
                hires_pipe,
                seeds,
                generators,
                images,
                pp,
            )

        if hasattr(self, "compel") and not pp.retain_compel_previous_load:
            del self.compel
        release_resources()

        yield images, [seeds, image_list, image_metadata]
