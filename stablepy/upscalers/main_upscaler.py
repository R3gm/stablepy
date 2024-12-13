import importlib

ANY_UPSCALER = (".pipelines.common", "UpscalerCommon")

UPSCALER_MAP = {
    # None: (".pipelines.base", "UpscalerNone"),
    # "None": (".pipelines.base", "UpscalerNone"),
    "Lanczos": (".pipelines.base", "UpscalerLanczos"),
    "Nearest": (".pipelines.base", "UpscalerNearest"),
    "ESRGAN_4x": (".pipelines.common", "UpscalerCommon"),
    "DAT x2": (".pipelines.common", "UpscalerCommon"),
    "DAT x3": (".pipelines.common", "UpscalerCommon"),
    "DAT x4": (".pipelines.common", "UpscalerCommon"),
    "HAT x4": (".pipelines.common", "UpscalerCommon"),
    "R-ESRGAN General 4xV3": (".pipelines.common", "UpscalerCommon"),
    "R-ESRGAN General WDN 4xV3": (".pipelines.common", "UpscalerCommon"),
    "R-ESRGAN AnimeVideo": (".pipelines.common", "UpscalerCommon"),
    "R-ESRGAN 4x+": (".pipelines.common", "UpscalerCommon"),
    "R-ESRGAN 4x+ Anime6B": (".pipelines.common", "UpscalerCommon"),
    "R-ESRGAN 2x+": (".pipelines.common", "UpscalerCommon"),
    "ScuNET GAN": (".pipelines.scunet", "UpscalerScuNET"),
    "ScuNET PSNR": (".pipelines.scunet", "UpscalerScuNET"),
    "SwinIR 4x": (".pipelines.swinir", "UpscalerSwinIR"),
}

BUILTIN_UPSCALERS = list(UPSCALER_MAP.keys())


def load_upscaler_model(**kwargs):
    model = kwargs.get("model", None)

    # Get the module and class model based on `model`
    module_path, class_name = UPSCALER_MAP.get(model, ANY_UPSCALER)

    # Import the module and get the class
    module = importlib.import_module(module_path, package=__package__)
    cls = getattr(module, class_name)

    return cls(**kwargs)
