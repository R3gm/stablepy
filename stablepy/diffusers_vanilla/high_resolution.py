from ..upscalers.esrgan import UpscalerESRGAN, UpscalerLanczos, UpscalerNearest
from ..logging.logging_setup import logger
import torch
import gc
from diffusers import DDIMScheduler
from diffusers.image_processor import VaeImageProcessor

latent_upscale_modes = {
    "Latent": {"mode": "bilinear", "antialias": False},
    "Latent (antialiased)": {"mode": "bilinear", "antialias": True},
    "Latent (bicubic)": {"mode": "bicubic", "antialias": False},
    "Latent (bicubic antialiased)": {"mode": "bicubic", "antialias": True},
    "Latent (nearest)": {"mode": "nearest", "antialias": False},
    "Latent (nearest-exact)": {"mode": "nearest-exact", "antialias": False},
}

LATENT_UPSCALERS = latent_upscale_modes.keys()


def process_images_high_resolution(
    images,
    upscaler_model_path, upscaler_increases_size,
    esrgan_tile=None, esrgan_tile_overlap=None,
    hires_steps=1, hires_params_config=None,
    task_name=None,
    generator=None,
    hires_pipe=None,
):

    def upscale_images(images, upscaler_model_path, esrgan_tile, esrgan_tile_overlap):
        if upscaler_model_path is not None:
            if upscaler_model_path == "Lanczos":
                scaler = UpscalerLanczos()
            elif upscaler_model_path == "Nearest":
                scaler = UpscalerNearest()
            else:
                scaler = UpscalerESRGAN(esrgan_tile, esrgan_tile_overlap)

            result_scaler = []
            for img_pre_up in images:
                image_pos_up = scaler.upscale(
                    img_pre_up, upscaler_increases_size, upscaler_model_path
                )
                torch.cuda.empty_cache()
                gc.collect()
                result_scaler.append(image_pos_up)
            images = result_scaler
            logger.info(f"Upscale resolution: {images[0].size}")

        return images

    def hires_fix(images):
        if hires_steps > 1:
            control_image_up = None
            if task_name not in ["txt2img", "inpaint", "img2img"]:
                control_image_up = images[0]
                images = images[1:]

            result_hires = []
            for img_pre_hires in images:
                try:
                    img_pos_hires = hires_pipe(
                        generator=generator,
                        image=img_pre_hires,
                        **hires_params_config,
                    ).images[0]
                except Exception as e:
                    e = str(e)
                    if "Tensor with 2 elements cannot be converted to Scalar" in e:
                        logger.debug(e)
                        logger.error("Error in sampler; trying with DDIM sampler")
                        hires_pipe.scheduler = DDIMScheduler.from_config(hires_pipe.scheduler.config)
                        img_pos_hires = hires_pipe(
                            generator=generator,
                            image=img_pre_hires,
                            **hires_params_config,
                        ).images[0]
                    elif "The size of tensor a (0) must match the size of tensor b (3) at non-singleton" in e or "cannot reshape tensor of 0 elements into shape [0, -1, 1, 512] because the unspecified dimensi" in e:
                        logger.error("Strength or steps too low for the model to produce a satisfactory response, returning image only with upscaling.")
                        img_pos_hires = img_pre_hires
                    else:
                        logger.error(e)
                        logger.error("The hiresfix couldn't be applied, returning image only with upscaling.")
                        img_pos_hires = img_pre_hires
                torch.cuda.empty_cache()
                gc.collect()
                result_hires.append(img_pos_hires)
            images = result_hires

            if control_image_up:
                images = [control_image_up] + images

        return images

    if upscaler_model_path in LATENT_UPSCALERS:
        control_image_up_pre_latent = None
        if task_name not in ["txt2img", "inpaint", "img2img"]:
            control_image_up_pre_latent = [images[0]]
            images = images[1:]

        image_processor = VaeImageProcessor()
        images_conversion = []
        for img_base in images:
            if not isinstance(img_base, torch.Tensor):
                prep_image = image_processor.preprocess(img_base)
                prep_image = prep_image.to(device=hires_pipe.vae.device.type, dtype=hires_pipe.vae.dtype)

                with torch.no_grad():
                    img_base = hires_pipe.vae.encode(prep_image).latent_dist.sample()

                img_base = hires_pipe.vae.config.scaling_factor * img_base

            images_conversion.append(img_base)

        config_latent = latent_upscale_modes[upscaler_model_path]

        logger.debug(str(images_conversion[0].shape))

        images = [
          torch.nn.functional.interpolate(
            im_l,
            size=(
                int(images_conversion[0].shape[2] * upscaler_increases_size),  # maybe round instead of int
                int(images_conversion[0].shape[3] * upscaler_increases_size),
            ),
            mode=config_latent["mode"],
            antialias=config_latent["antialias"],
            ) for im_l in images_conversion
        ]

        logger.debug(str(images[0].shape))
        logger.info(
            "Latent resolution: "
            f"{images[0].shape[2] * 8}x{images[0].shape[3] * 8}"
        )

        if control_image_up_pre_latent:
            images = control_image_up_pre_latent + images

        torch.cuda.empty_cache()

    else:
        images = upscale_images(
            images, upscaler_model_path, esrgan_tile, esrgan_tile_overlap
        )

    images = hires_fix(images)

    return images
