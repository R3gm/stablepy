from diffusers import (
    MotionAdapter,
    AnimateDiffPipeline,
    AutoPipelineForImage2Image,
    StableDiffusionXLPipeline,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionXLInpaintPipeline,
    ControlNetModel,
    StableDiffusionPipeline
)
import torch
from ..logging.logging_setup import logger


def custom_task_model_loader(
    pipe,
    model_category="detailfix",
    task_name="txt2img",
    torch_dtype=torch.float16
):
    # Pipe detailfix_pipe
    if model_category == "detailfix":

        if hasattr(pipe, "transformer"):
            from .extra_pipe.flux.pipeline_flux_inpaint import FluxInpaintPipeline
            return FluxInpaintPipeline(
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                scheduler=pipe.scheduler,
                text_encoder_2=pipe.text_encoder_2,
                tokenizer_2=pipe.tokenizer_2,
                transformer=pipe.transformer,
            )

        if not hasattr(pipe, "text_encoder_2"):
            # sd df
            if torch_dtype == torch.float16:
                type_params = {"torch_dtype": torch.float16, "variant": "fp16"}
            else:
                type_params = {"torch_dtype": torch.float32}
            logger.debug(f"Params detailfix sd controlnet {type_params}")
            controlnet_detailfix = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_inpaint", **type_params,
            )
            detailfix_pipe = StableDiffusionControlNetInpaintPipeline(
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                unet=pipe.unet,
                controlnet=controlnet_detailfix,
                scheduler=pipe.scheduler,
                safety_checker=pipe.safety_checker,
                feature_extractor=pipe.feature_extractor,
                image_encoder=pipe.image_encoder,
                requires_safety_checker=pipe.config.requires_safety_checker,
            )
        else:
            # sdxl df
            detailfix_pipe = StableDiffusionXLInpaintPipeline(
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                text_encoder_2=pipe.text_encoder_2,
                tokenizer=pipe.tokenizer,
                tokenizer_2=pipe.tokenizer_2,
                unet=pipe.unet,
                # controlnet=controlnet,
                scheduler=pipe.scheduler,
                feature_extractor=pipe.feature_extractor,
                image_encoder=pipe.image_encoder,
            )
            detailfix_pipe.enable_vae_slicing()
            detailfix_pipe.enable_vae_tiling()
            detailfix_pipe.watermark = None

        return detailfix_pipe

    elif model_category in ["hires", "detailfix_img2img"]:

        if hasattr(pipe, "transformer"):
            from .extra_pipe.flux.pipeline_flux_img2img import FluxImg2ImgPipeline
            return FluxImg2ImgPipeline(
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                scheduler=pipe.scheduler,
                text_encoder_2=pipe.text_encoder_2,
                tokenizer_2=pipe.tokenizer_2,
                transformer=pipe.transformer,
            )

        # Pipe hires detailfix_pipe img2img
        if task_name != "txt2img" or hasattr(pipe, "set_pag_applied_layers"):
            if not hasattr(pipe, "text_encoder_2"):
                hires_pipe = StableDiffusionPipeline(
                    vae=pipe.vae,
                    text_encoder=pipe.text_encoder,
                    tokenizer=pipe.tokenizer,
                    unet=pipe.unet,
                    scheduler=pipe.scheduler,
                    safety_checker=pipe.safety_checker,
                    feature_extractor=pipe.feature_extractor,
                    image_encoder=pipe.image_encoder,
                    requires_safety_checker=pipe.config.requires_safety_checker,
                )

            else:
                hires_pipe = StableDiffusionXLPipeline(
                    vae=pipe.vae,
                    text_encoder=pipe.text_encoder,
                    text_encoder_2=pipe.text_encoder_2,
                    tokenizer=pipe.tokenizer,
                    tokenizer_2=pipe.tokenizer_2,
                    unet=pipe.unet,
                    scheduler=pipe.scheduler,
                    feature_extractor=pipe.feature_extractor,
                    image_encoder=pipe.image_encoder,
                )

            hires_pipe = AutoPipelineForImage2Image.from_pipe(hires_pipe, enable_pag=False)
        else:
            hires_pipe = AutoPipelineForImage2Image.from_pipe(pipe, enable_pag=False)

        if hasattr(hires_pipe, "text_encoder_2"):
            hires_pipe.enable_vae_slicing()
            hires_pipe.enable_vae_tiling()
            hires_pipe.watermark = None

        return hires_pipe

    elif model_category == "animatediff":
        # Pipe animatediff
        if not hasattr(pipe, "text_encoder_2"):
            adapter = MotionAdapter.from_pretrained(
                "guoyww/animatediff-motion-adapter-v1-5-2"
            )
            adapter.to("cuda" if torch.cuda.is_available() else "cpu")

            animatediff_pipe = AnimateDiffPipeline(
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                unet=pipe.unet,
                motion_adapter=adapter,
                scheduler=pipe.scheduler,
                feature_extractor=pipe.feature_extractor,
                image_encoder=pipe.image_encoder,
            )
        else:
            raise ValueError("Animatediff not implemented for SDXL")

        return animatediff_pipe
