import os
from abc import abstractmethod
import PIL
from PIL import Image

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
NEAREST = (Image.Resampling.NEAREST if hasattr(Image, 'Resampling') else Image.NEAREST)


class Upscaler:
    name = None
    model_path = None
    model_name = None
    model_url = None
    model = None
    scalers: list
    tile = True

    def __init__(self):
        self.tile_size = 192
        self.tile_pad = 8
        self.device = "cpu"
        self.scale = 1
        self.half = False
        self.model_download_path = os.path.join(os.path.expanduser("~"), ".cache", "upscalers")
        self.can_tile = True
        self.disable_progress_bar = False

    @abstractmethod
    def do_upscale(self, img: PIL.Image):
        return img

    def upscale(self, img: PIL.Image, scale, disable_progress_bar=False):
        self.disable_progress_bar = disable_progress_bar
        self.scale = scale

        dest_w = int((img.width * scale) // 8 * 8)
        dest_h = int((img.height * scale) // 8 * 8)

        for i in range(3):
            if img.width >= dest_w and img.height >= dest_h and (i > 0 or scale != 1):
                break

            shape = (img.width, img.height)

            img = self.do_upscale(img)

            if shape == (img.width, img.height):
                break

        if img.width != dest_w or img.height != dest_h:
            img = img.resize((int(dest_w), int(dest_h)), resample=LANCZOS)

        return img

    @abstractmethod
    def load_model(self, path: str):
        pass


class UpscalerData:
    name = None
    data_path = None
    scale: int = 4
    scaler: Upscaler = None
    model: None

    def __init__(self, name: str, path: str, upscaler: Upscaler = None, scale: int = 4, supports_half=True, model=None):
        self.name = name
        self.data_path = path
        self.local_data_path = path
        self.scaler = upscaler
        self.scale = scale
        self.supports_half = supports_half
        self.model = model

    def __repr__(self):
        return f"<UpscalerData name={self.name} data_path={self.data_path} scale={self.scale}>"


class UpscalerNone(Upscaler):
    name = "None"
    scalers = []

    def load_model(self, path):
        pass

    def do_upscale(self, img):
        return img

    def __init__(self, **kwargs):
        super().__init__()
        self.scalers = [UpscalerData("None", None, self)]


class UpscalerLanczos(Upscaler):
    scalers = []

    def do_upscale(self, img):
        return img.resize((int(img.width * self.scale), int(img.height * self.scale)), resample=LANCZOS)

    def load_model(self, _):
        pass

    def __init__(self, **kwargs):
        super().__init__()
        self.name = "Lanczos"
        self.scalers = [UpscalerData("Lanczos", None, self)]


class UpscalerNearest(Upscaler):
    scalers = []

    def do_upscale(self, img):
        return img.resize((int(img.width * self.scale), int(img.height * self.scale)), resample=NEAREST)

    def load_model(self, _):
        pass

    def __init__(self, **kwargs):
        super().__init__()
        self.name = "Nearest"
        self.scalers = [UpscalerData("Nearest", None, self)]


if __name__ == "__main__":
    up = UpscalerNearest()
    scale_up = 1.1
    img = Image.open("img.png")
    print(img.size)
    img_up = up.upscale(img, scale_up)
    print(img_up.size)
