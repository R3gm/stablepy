import numpy as np
from ..utils import release_resources
import PIL
from .image_utils import (
    HWC3,
    resize_image,
    apply_gaussian_blur,
    recolor_luminance,
    recolor_intensity,
)
from .constans_preprocessor import (
    AUX_TASKS,
    TRANSFORMERS_LIB_TASKS,
    AUX_BETA_TASKS,
    TASK_AND_PREPROCESSORS,
)


def process_basic_task(image: np.ndarray, resolution: int) -> PIL.Image.Image:
    """Process basic tasks that require only resizing."""
    image = HWC3(image)
    image = resize_image(image, resolution=resolution)
    return PIL.Image.fromarray(image)


def process_tile_task(image: np.ndarray, resolution: int, preprocessor_name: str) -> PIL.Image.Image:
    """Process the 'tile' task with Gaussian blur."""
    blur_levels = {"Mild Blur": 3, "Moderate Blur": 15, "Heavy Blur": 27}
    if preprocessor_name not in blur_levels:
        raise ValueError(f"Invalid preprocessor name for tile task: {preprocessor_name}")
    image = resize_image(image, resolution=resolution)
    image = apply_gaussian_blur(image, ksize=blur_levels[preprocessor_name])
    return PIL.Image.fromarray(image)


def process_recolor_task(image: np.ndarray, resolution: int, preprocessor_name: str, gamma_correction: float) -> PIL.Image.Image:
    """Process the 'recolor' task."""
    image = resize_image(image, resolution=resolution)
    if preprocessor_name == "Recolor luminance":
        image = recolor_luminance(image, thr_a=gamma_correction)
    elif preprocessor_name == "Recolor intensity":
        image = recolor_intensity(image, thr_a=gamma_correction)
    else:
        raise ValueError(f"Invalid preprocessor name for recolor task: {preprocessor_name}")
    return PIL.Image.fromarray(image)


class Preprocessor:
    MODEL_ID = "lllyasviel/Annotators"

    def __init__(self):
        self.model = None
        self.name = ""

    def _load_aux_model(self, name: str):
        """Lazy load models from the `controlnet_aux` library."""
        import controlnet_aux as cnx

        model_map = {
            "HED": lambda: cnx.HEDdetector.from_pretrained(self.MODEL_ID),
            "Midas": lambda: cnx.MidasDetector.from_pretrained(self.MODEL_ID),
            "MLSD": lambda: cnx.MLSDdetector.from_pretrained(self.MODEL_ID),
            "Openpose": lambda: cnx.OpenposeDetector.from_pretrained(self.MODEL_ID),
            "PidiNet": lambda: cnx.PidiNetDetector.from_pretrained(self.MODEL_ID),
            "NormalBae": lambda: cnx.NormalBaeDetector.from_pretrained(self.MODEL_ID),
            "Lineart": lambda: cnx.LineartDetector.from_pretrained(self.MODEL_ID),
            "LineartAnime": lambda: cnx.LineartAnimeDetector.from_pretrained(self.MODEL_ID),
            "Canny": lambda: cnx.CannyDetector(),
            "ContentShuffle": lambda: cnx.ContentShuffleDetector(),
        }

        if name in model_map:
            return model_map[name]()

        raise ValueError(f"Unsupported task name: {name}")

    def _load_transformers_model(self, name: str):
        """Lazy load models from the `.transformers_lib.pipelines`."""
        from .transformers_lib.pipelines import (
            DPTDepthEstimator,
            UP_ImageSegmentor,
            ZoeDepth,
            SegFormer,
            DepthAnything,
        )
        model_map = {
            TRANSFORMERS_LIB_TASKS[0]: DPTDepthEstimator,
            TRANSFORMERS_LIB_TASKS[1]: UP_ImageSegmentor,
            TRANSFORMERS_LIB_TASKS[2]: ZoeDepth,
            TRANSFORMERS_LIB_TASKS[3]: SegFormer,
            TRANSFORMERS_LIB_TASKS[4]: DepthAnything,
        }
        if name in model_map:
            return model_map[name]()
        raise ValueError(f"Unsupported task name: {name}")

    def _load_custom_model(self, name: str):
        """Lazy load custom models from specialized modules."""
        if name == AUX_BETA_TASKS[0]:
            from .controlnet_aux_beta.teed import TEEDdetector
            return TEEDdetector()
        elif name == AUX_BETA_TASKS[1]:
            from .controlnet_aux_beta.anyline import AnylineDetector
            return AnylineDetector()
        elif name == AUX_BETA_TASKS[2]:
            from .controlnet_aux_beta.lineart_standard import LineartStandardDetector
            return LineartStandardDetector()
        raise ValueError(f"Unsupported task name: {name}")

    def load(self, name: str, use_cuda: bool = False) -> None:
        """Load the specified preprocessor model."""
        if name == self.name:
            if use_cuda and hasattr(self.model, "to"):
                self.model.to("cuda")
            return  # Skip if already loaded

        if name in AUX_TASKS:
            self.model = self._load_aux_model(name)
        elif name in TRANSFORMERS_LIB_TASKS:
            self.model = self._load_transformers_model(name)
        elif name in AUX_BETA_TASKS:
            self.model = self._load_custom_model(name)
        else:
            raise ValueError(f"Unknown preprocessor name: {name}")

        release_resources()

        self.name = name
        if use_cuda and hasattr(self.model, "to"):
            self.model.to("cuda")

    def __call__(self, image: PIL.Image.Image, **kwargs) -> PIL.Image.Image:
        """Process an image using the loaded preprocessor model."""
        if not self.model:
            raise RuntimeError("No model is loaded. Please call `load()` first.")

        if self.name == "Canny":
            return self._process_canny(image, **kwargs)
        elif self.name == "Midas":
            return self._process_midas(image, **kwargs)
        else:
            return self.model(image, **kwargs)

    def _process_canny(self, image: PIL.Image.Image, **kwargs) -> PIL.Image.Image:
        """Process an image using the Canny preprocessor."""
        detect_resolution = kwargs.pop("detect_resolution", None)
        image = np.array(image)
        image = HWC3(image)
        if detect_resolution:
            image = resize_image(image, resolution=detect_resolution)
        image = self.model(image, **kwargs)
        return PIL.Image.fromarray(image)

    def _process_midas(self, image: PIL.Image.Image, **kwargs) -> PIL.Image.Image:
        """Process an image using the Midas preprocessor."""
        detect_resolution = kwargs.pop("detect_resolution", 512)
        image_resolution = kwargs.pop("image_resolution", 512)
        image = np.array(image)
        image = HWC3(image)
        image = resize_image(image, resolution=detect_resolution)
        image = self.model(image, **kwargs)
        image = HWC3(image)
        image = resize_image(image, resolution=image_resolution)
        return PIL.Image.fromarray(image)


def get_preprocessor_params(
    image: np.ndarray,
    task_name: str,
    preprocessor_name: str,
    image_resolution: int,
    preprocess_resolution: int,
    low_threshold: int,
    high_threshold: int,
    value_threshold: float,
    distance_threshold: float,
) -> tuple[dict, str]:
    """
    Determine the parameters and model name for preprocessing.

    Args:
        image (np.ndarray): The input image.
        task_name (str): The name of the task.
        preprocessor_name (str): The name of the preprocessor.
        image_resolution (int): The resolution of the input image.
        preprocess_resolution (int): The resolution to preprocess to.
        low_threshold (int): Low threshold for edge detection.
        high_threshold (int): High threshold for edge detection.
        value_threshold (float): Threshold for MLSD value detection.
        distance_threshold (float): Threshold for MLSD distance detection.

    Returns:
        tuple[dict, str]: A dictionary of parameters for preprocessing and the model name.
    """
    params_preprocessor = {
        "image": image,
        "image_resolution": image_resolution,
        "detect_resolution": preprocess_resolution,
    }
    model_name = None

    if task_name in ["canny", "sdxl_canny_t2i"]:
        params_preprocessor.update({
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
        })
        model_name = "Canny"
    elif task_name in ["openpose", "sdxl_openpose_t2i"]:
        params_preprocessor["hand_and_face"] = not ("core" in preprocessor_name)
        model_name = "Openpose"
    elif task_name in ["depth", "sdxl_depth-midas_t2i"]:
        model_name = preprocessor_name
    elif task_name == "mlsd":
        params_preprocessor.update({
            "thr_v": value_threshold,
            "thr_d": distance_threshold,
        })
        model_name = "MLSD"
    elif task_name in ["scribble", "sdxl_sketch_t2i"]:
        if "HED" in preprocessor_name:
            params_preprocessor["scribble"] = False
            model_name = "HED"
        elif "TEED" in preprocessor_name:
            model_name = "TEED"
        else:
            params_preprocessor["safe"] = False
            model_name = "PidiNet"
    elif task_name == "softedge":
        if "HED" in preprocessor_name:
            params_preprocessor["scribble"] = "safe" in preprocessor_name
            model_name = "HED"
        elif "TEED" in preprocessor_name:
            model_name = "TEED"
        else:
            params_preprocessor["safe"] = "safe" in preprocessor_name
            model_name = "PidiNet"
    elif task_name == "segmentation":
        model_name = preprocessor_name
    elif task_name == "normalbae":
        model_name = "NormalBae"
    elif task_name in ["lineart", "lineart_anime", "sdxl_lineart_t2i"]:
        if preprocessor_name in ["Lineart standard", "Anyline"]:
            model_name = preprocessor_name
        else:
            model_name = "LineartAnime" if "anime" in preprocessor_name.lower() else "Lineart"
            if "coarse" in preprocessor_name:
                params_preprocessor["coarse"] = "coarse" in preprocessor_name
    elif task_name == "shuffle":
        params_preprocessor.pop("detect_resolution", None)
        model_name = preprocessor_name

    return params_preprocessor, model_name
