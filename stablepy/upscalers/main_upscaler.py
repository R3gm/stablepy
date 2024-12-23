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
    """
    Loads and returns an upscaler model class instance based on the provided keyword arguments.

    Keyword Args:
        model (str): The name or path of the model to load. It can be any of the BUILTIN_UPSCALERS.
        tile (int, optional): The size of the tiles to use for upscaling. Default is 192.
        tile_overlap (int, optional): The overlap between tiles. Default is 8.
        device (str, optional): The device to use for computation, e.g., "cuda" or "cpu". Default is "cuda".
        half (bool, optional): Whether to use half-precision floats. Default is False.
        **kwargs: Additional keyword arguments to pass to the model class constructor.

    Returns:
        object: An instance of the upscaler model class.

    Example:
        from PIL import Image

        # Load the upscaler model
        upscaler = load_upscaler_model(model="your_model_name_or_path", tile=192, tile_overlap=8, device="cuda", half=False)

        # Open an image using PIL
        img_pre_up = Image.open("path_to_your_image.jpg")

        # Define the upscaling parameters
        upscaler_increases_size = 1.4
        disable_progress_bar = False

        # Use the upscaler to upscale the image
        image_pos_up = upscaler.upscale(img_pre_up, upscaler_increases_size, disable_progress_bar)
    """

    model = kwargs.get("model", None)

    # Get the module and class model based on `model`
    module_path, class_name = UPSCALER_MAP.get(model, ANY_UPSCALER)

    # Import the module and get the class
    module = importlib.import_module(module_path, package=__package__)
    cls = getattr(module, class_name)

    return cls(**kwargs)
