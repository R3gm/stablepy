from ..utils_upscaler import upscale_2, load_spandrel_model, release_resources_upscaler, load_file_from_url
from .base import Upscaler, UpscalerData
from PIL import Image

SWINIR_MODEL_URL = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"


class UpscalerSwinIR(Upscaler):
    def __init__(self, model="SwinIR 4x", tile=192, tile_overlap=8, device="cuda", half=False, **kwargs):
        self._cached_model = None
        self._cached_model_config = None
        self.name = "SwinIR"
        self.model_url = SWINIR_MODEL_URL
        self.model_name = "SwinIR 4x"
        super().__init__()
        self.scalers = [UpscalerData(self.model_name, self.model_url, self)]

        self.device = device
        self.half = half
        self.tile = tile
        self.tile_overlap = tile_overlap

        release_resources_upscaler()

        try:
            self.model_descriptor = self.load_model(model)
        except Exception as e:
            print(f"Unable to load SwinIR model {model}: {e}")
            self.model_descriptor = None

    def do_upscale(self, img: Image.Image) -> Image.Image:
        release_resources_upscaler()

        if self.model_descriptor is None:
            return img

        img = upscale_2(
            img,
            self.model_descriptor,
            tile_size=self.tile,
            tile_overlap=self.tile_overlap,
            scale=self.model_descriptor.scale,
            desc="SwinIR",
            disable_progress_bar=self.disable_progress_bar,
        )

        release_resources_upscaler()

        return img

    def load_model(self, path, scale=4):
        if self.scalers[0].name == path:
            path = self.scalers[0].data_path

        if path.startswith("http"):
            filename = load_file_from_url(
                url=path,
                model_dir=self.model_download_path,
                file_name=f"{self.model_name.replace(' ', '_')}.pth",
            )
        else:
            filename = path

        model_descriptor = load_spandrel_model(
            filename,
            device=self.device,
            # prefer_half=self.half,
            expected_architecture="SwinIR",
        )

        # try:
        #     model_descriptor.model.compile()
        # except Exception:
        #     logger.warning("Failed to compile SwinIR model, fallback to JIT")

        return model_descriptor


if __name__ == "__main__":
    from PIL import Image

    up = UpscalerSwinIR(model="SwinIR 4x", tile=192, tile_overlap=8, device="cuda", half=False)
    scale_up = 1.1
    img = Image.open("img.png")
    print(img.size)
    img_up = up.upscale(img, scale_up)
    print(img_up.size)
