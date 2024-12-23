# Code based based from the repository comfyui_controlnet_aux:
# https://github.com/Fannovel16/comfyui_controlnet_aux/blob/main/src/controlnet_aux/lineart_standard/__init__.py
import cv2
import numpy as np
from PIL import Image
from ..preprocessor_utils import HWC3, fast_resize_image


class LineartStandardDetector:
    def __call__(
        self,
        image=None,
        guassian_sigma=6.0,
        intensity_threshold=8,
        **kwargs
    ):

        detect_resolution = kwargs.pop("detect_resolution", 512)
        image_resolution = kwargs.pop("image_resolution", 512)

        image = HWC3(image)
        original_height, original_width, _ = image.shape
        image = fast_resize_image(image, detect_resolution)

        x = image.astype(np.float32)
        g = cv2.GaussianBlur(x, (0, 0), guassian_sigma)
        intensity = np.min(g - x, axis=2).clip(0, 255)
        intensity /= max(16, np.median(intensity[intensity > intensity_threshold]))
        intensity *= 127
        detected_map = intensity.clip(0, 255).astype(np.uint8)

        detected_map = HWC3(detected_map)
        resize_result = fast_resize_image(
            detected_map, image_resolution
        )

        return Image.fromarray(resize_result)
