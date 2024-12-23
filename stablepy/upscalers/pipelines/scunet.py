from ..utils_upscaler import upscale_2, load_spandrel_model, release_resources_upscaler, load_file_from_url
from .base import Upscaler, UpscalerData
import PIL.Image
import os


class UpscalerScuNET(Upscaler):
    def __init__(self, model="ScuNET GAN", tile=192, tile_overlap=8, device="cuda", half=False, **kwargs):
        self.name = "ScuNET"
        self.model_name = "ScuNET GAN"
        self.model_name2 = "ScuNET PSNR"
        self.model_url = "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_gan.pth"
        self.model_url2 = "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth"
        super().__init__()
        model_a = UpscalerData(self.model_name, self.model_url, self, 4)
        model_b = UpscalerData(self.model_name2, self.model_url2, self)
        self.scalers = [model_a, model_b]

        self.device = device
        self.half = half
        self.tile = tile
        self.tile_overlap = tile_overlap

        release_resources_upscaler()

        try:
            self.model_descriptor = self.load_model(model)
        except Exception as e:
            print(f"Unable to load ScuNET model {model}: {e}")
            self.model_descriptor = None

    def do_upscale(self, img: PIL.Image.Image):
        release_resources_upscaler()

        if self.model_descriptor is None:
            return img

        img = upscale_2(
            img,
            self.model_descriptor,
            tile_size=self.tile,
            tile_overlap=self.tile_overlap,
            scale=1,  # ScuNET is a denoising model, not an upscaler
            desc='ScuNET',
            disable_progress_bar=self.disable_progress_bar,
        )

        release_resources_upscaler()

        return img

    def load_model(self, path: str):
        for scaler in self.scalers:
            if scaler.name == path:
                if scaler.local_data_path.startswith("http"):
                    scaler.local_data_path = load_file_from_url(
                        scaler.data_path,
                        model_dir=self.model_download_path,
                    )
                if not os.path.exists(scaler.local_data_path):
                    raise FileNotFoundError(f"ScuNET data missing: {scaler.local_data_path}")
                return load_spandrel_model(
                    scaler.local_data_path,
                    device=self.device,
                    # prefer_half=self.half,
                    expected_architecture="SCUNet",
                )
        raise ValueError(f"Unable to find model info: {path}")


if __name__ == "__main__":
    from PIL import Image

    up = UpscalerScuNET(model="ScuNET PSNR", tile=192, tile_overlap=8, device="cuda", half=False)
    scale_up = 1.1
    img = Image.open("img.png")
    print(img.size)
    img_up = up.upscale(img, scale_up)  # ScuNET PSNR ScuNET GAN
    print(img_up.size)
