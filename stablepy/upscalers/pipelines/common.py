from ..utils_upscaler import load_spandrel_model, upscale_with_model, release_resources_upscaler, load_file_from_url
from .base import Upscaler, UpscalerData
import os


class UpscalerCommon(Upscaler):
    def __init__(self, model="R-ESRGAN 4x+", tile=192, tile_overlap=8, device="cuda", half=False, **kwargs):
        self.name = "RealESRGAN"
        super().__init__()
        self.scalers = get_models(self)

        self.device = device
        self.half = half
        self.tile = tile
        self.tile_overlap = tile_overlap

        release_resources_upscaler()

        try:
            self.model_descriptor = self.load_model(model)
        except Exception as e:
            print(f"Unable to load upscaler model {model}: {e}")
            self.model_descriptor = None

    def do_upscale(self, img):
        release_resources_upscaler()

        if self.model_descriptor is None:
            return img

        return upscale_with_model(
            self.model_descriptor,
            img,
            tile_size=self.tile,
            tile_overlap=self.tile_overlap,
            # TODO: `outscale`?
            disable_progress_bar=self.disable_progress_bar,
        )

    def load_model(self, path):
        for scaler in self.scalers:
            if scaler.name == path:
                if scaler.local_data_path.startswith("http"):
                    scaler.local_data_path = load_file_from_url(
                        scaler.data_path,
                        model_dir=self.model_download_path,
                    )
                if not os.path.exists(scaler.local_data_path):
                    raise FileNotFoundError(f"Upscaler model data missing: {scaler.local_data_path}")
                return load_spandrel_model(
                    scaler.local_data_path,
                    device=self.device,
                    prefer_half=self.half if scaler.supports_half else False,
                )

        # Load custom model
        if path.startswith("http"):
            filename = load_file_from_url(
                url=path,
                model_dir=self.model_download_path,
            )
        else:
            filename = path

        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Model file {filename} not found")

        return load_spandrel_model(
            filename,
            device=self.device,
            prefer_half=self.half,
        )


def get_models(scaler: UpscalerCommon):
    return [
        # ESRGAN
        UpscalerData(
            name="ESRGAN_4x",
            path="https://github.com/cszn/KAIR/releases/download/v1.0/ESRGAN.pth",
            scale=4,
            upscaler=scaler,
        ),
        # R-ESRGAN
        UpscalerData(
            name="R-ESRGAN General 4xV3",
            path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
            scale=4,
            upscaler=scaler,
        ),
        UpscalerData(
            name="R-ESRGAN General WDN 4xV3",
            path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
            scale=4,
            upscaler=scaler,
        ),
        UpscalerData(
            name="R-ESRGAN AnimeVideo",
            path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
            scale=4,
            upscaler=scaler,
        ),
        UpscalerData(
            name="R-ESRGAN 4x+",
            path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            scale=4,
            upscaler=scaler,
        ),
        UpscalerData(
            name="R-ESRGAN 4x+ Anime6B",
            path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            scale=4,
            upscaler=scaler,
        ),
        UpscalerData(
            name="R-ESRGAN 2x+",
            path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            scale=2,
            upscaler=scaler,
        ),
        # DAT
        UpscalerData(
            name="DAT x2",
            path="https://huggingface.co/w-e-w/DAT/resolve/main/experiments/pretrained_models/DAT/DAT_x2.pth",
            scale=2,
            upscaler=scaler,
            supports_half=False,
        ),
        UpscalerData(
            name="DAT x3",
            path="https://huggingface.co/w-e-w/DAT/resolve/main/experiments/pretrained_models/DAT/DAT_x3.pth",
            scale=3,
            upscaler=scaler,
            supports_half=False,
        ),
        UpscalerData(
            name="DAT x4",
            path="https://huggingface.co/w-e-w/DAT/resolve/main/experiments/pretrained_models/DAT/DAT_x4.pth",
            scale=4,
            upscaler=scaler,
            supports_half=False,
        ),
        # HAT
        UpscalerData(
            name="HAT x4",
            path="https://huggingface.co/Phips/4xNomos8kSCHAT-S/resolve/main/4xNomos8kSCHAT-S.safetensors",
            scale=4,
            upscaler=scaler,
            supports_half=False,
        ),
    ]


if __name__ == "__main__":
    from PIL import Image

    up = UpscalerCommon(model="R-ESRGAN 2x+", tile=192, tile_overlap=8, device="cuda", half=False)
    scale_up = 1.1
    img = Image.open("img.png")
    print(img.size)
    img_up = up.upscale(img, scale_up)
    print(img_up.size)
