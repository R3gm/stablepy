import numpy as np
import PIL
import torch

from ..utils import release_resources
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
    EXTRA_AUX_TASKS,
    TASK_AND_PREPROCESSORS,
)
from ..utils import convert_image_to_numpy_array


def process_basic_task(image: np.ndarray, resolution: int) -> PIL.Image.Image:
    """Process basic tasks that require only resizing."""
    image = HWC3(image)
    image = resize_image(image, resolution=resolution)
    return PIL.Image.fromarray(image)


class RecolorDetector:
    def resize(self, image, res):
        if image is None:
            raise ValueError("image must be defined.")

        image = HWC3(image)
        return resize_image(image, res)

    def __call__(self, image=None, gamma_correction=1.0, image_resolution=512, mode="luminance", **kwargs):
        """Process the 'recolor' task."""
        if mode == "luminance":
            func_c = recolor_luminance
        elif mode == "intensity":
            func_c = recolor_intensity
        else:
            raise ValueError("Invalid recolor mode")

        result = func_c(
            self.resize(image, image_resolution), thr_a=gamma_correction
        )
        return PIL.Image.fromarray(HWC3(result))


class BlurDetector(RecolorDetector):
    def __call__(self, image=None, image_resolution=512, blur_sigma=5, **kwargs):
        """Process the 'tile' task with Gaussian blur."""
        result = apply_gaussian_blur(
            self.resize(image, image_resolution), ksize=blur_sigma
        )
        return PIL.Image.fromarray(HWC3(result))


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

    def _load_extra_model(self, name: str):
        """Lazy load custom models from specialized modules."""
        if name == EXTRA_AUX_TASKS[0]:
            return RecolorDetector()
        elif name == EXTRA_AUX_TASKS[1]:
            return BlurDetector()
        raise ValueError(f"Unsupported task name: {name}")

    def to(self, device):
        if hasattr(self.model, "to"):
            self.model.to(device)

    def load(self, name: str, use_cuda: bool = False) -> None:
        """
        Load the specified preprocessor model.
        Parameters:
        name (str): The name of the preprocessor model to load.
        use_cuda (bool, optional): If True, the model will be moved to GPU. Defaults to False.
        Raises:
        ValueError: If the specified preprocessor name is not recognized.
        Notes:
        - If the specified model is already loaded, the function will return early.
        - The function will release any previously held resources before loading a new model.
        - The model can be loaded from different sources based on the name provided.
        """

        if name == self.name:
            if use_cuda:
                self.to("cuda")
            return  # Skip if already loaded

        if name in AUX_TASKS:
            self.model = self._load_aux_model(name)
        elif name in TRANSFORMERS_LIB_TASKS:
            self.model = self._load_transformers_model(name)
        elif name in AUX_BETA_TASKS:
            self.model = self._load_custom_model(name)
        elif name in EXTRA_AUX_TASKS:
            self.model = self._load_extra_model(name)
        else:
            raise ValueError(f"Unknown preprocessor name: {name}")

        release_resources()

        self.name = name

        if use_cuda:
            self.to("cuda")

    def __call__(self, image: PIL.Image.Image, **kwargs) -> PIL.Image.Image:
        """
        Process an image using the preprocessor function or model.

        Args:
            image (PIL.Image.Image): The input image.
            **kwargs: Additional parameters for preprocessing, which may include:
                - image_resolution (int): The proportional maximum resolution of the input image while preserving the aspect ratio (applicable for all tasks).
                - detect_resolution (int): The resolution to preprocess to (applicable for all tasks).
                - low_threshold (int): Low threshold for edge detection (applicable for Canny).
                - high_threshold (int): High threshold for edge detection (applicable for Canny).
                - thr_v (float): Threshold for MLSD value detection (applicable for MLSD).
                - thr_d (float): Threshold for MLSD distance detection (applicable for MLSD).
                - mode (str): Mode for Recolor (e.g., 'intensity' or 'luminance') (applicable for Recolor).
                - gamma_correction (float): Gamma correction value for Recolor (applicable for Recolor).
                - blur_sigma (int): Sigma value for Blur (applicable for Blur).
                - hand_and_face (bool): Whether to include hand and face detection (applicable for Openpose).
                - scribble (bool): Whether to use scribble mode (applicable for HED).
                - safe (bool): Whether to use safe mode (applicable for PidiNet).
                - coarse (bool): Whether to use coarse mode (applicable for Lineart).

        Returns:
            PIL.Image.Image: The processed image.
        """

        if not self.model:
            raise RuntimeError("No model is loaded. Please call `load()` first.")

        if not isinstance(image, np.ndarray):
            image = convert_image_to_numpy_array(image)

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
        image = self.model(image)  # , **kwargs)
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
    gamma_correction: float,
    blur_sigma: int,
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
        gamma_correction (float): Threshold for Recolor thr_a.
        blur_sigma (int): Threshold for Blur sigma.

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
    elif task_name == "recolor":
        if "intensity" in preprocessor_name:
            params_preprocessor["mode"] = "intensity"
        else:
            params_preprocessor["mode"] = "luminance"
        params_preprocessor["gamma_correction"] = gamma_correction
        model_name = "Recolor"
    elif task_name == "tile":
        params_preprocessor["blur_sigma"] = blur_sigma
        model_name = "Blur"

    return params_preprocessor, model_name
