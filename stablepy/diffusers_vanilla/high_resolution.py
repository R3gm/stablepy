from ..upscalers.esrgan import UpscalerESRGAN, UpscalerLanczos, UpscalerNearest
from ..logging.logging_setup import logger
import torch, gc

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
        if upscaler_model_path != None:
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
            if task_name not in ["txt2img", "inpaint", "img2img"]:
                control_image_up = images[0]
                images = images[1:]

            result_hires = []
            for img_pre_hires in images:
                img_pos_hires = hires_pipe(
                    generator=generator,
                    image=img_pre_hires,
                    **hires_params_config,
                ).images[0]
                torch.cuda.empty_cache()
                gc.collect()
                result_hires.append(img_pos_hires)
            images = result_hires

            if task_name not in ["txt2img", "inpaint", "img2img"]:
                images = [control_image_up] + images
        return images

    images = upscale_images(images, upscaler_model_path, esrgan_tile, esrgan_tile_overlap)
    images = hires_fix(images)

    return images
