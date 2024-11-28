import os
import cv2
import numpy as np
import torch
from einops import rearrange
from huggingface_hub import hf_hub_download
from PIL import Image
from ..preprocessor_utils import HWC3, fast_resize_image, safe_step
from .ted import TED


class TEEDdetector:
    def __init__(self, pretrained_model_or_path=None, filename=None, subfolder=None):
        if not pretrained_model_or_path:
            pretrained_model_or_path = "fal-ai/teed"
            filename = "5_model.pth"
            subfolder = None
        if os.path.isdir(pretrained_model_or_path):
            model_path = os.path.join(pretrained_model_or_path, filename)
        else:
            model_path = hf_hub_download(
                pretrained_model_or_path, filename, subfolder=subfolder
            )

        model = TED()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model = model

    def to(self, device):
        self.model.to(device)
        return self

    def __call__(
        self,
        image,
        safe_steps=2,
        **kwargs
    ):
        detect_resolution = kwargs.pop("detect_resolution", 512)
        image_resolution = kwargs.pop("image_resolution", 512)

        device = next(iter(self.model.parameters())).device

        image = HWC3(image)
        original_height, original_width, _ = image.shape
        image = fast_resize_image(image, detect_resolution)

        assert image.ndim == 3
        height, width, _ = image.shape
        with torch.no_grad():
            image_teed = torch.from_numpy(image.copy()).float().to(device)
            image_teed = rearrange(image_teed, "h w c -> 1 c h w")
            edges = self.model(image_teed)
            edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
            edges = [
                cv2.resize(e, (width, height), interpolation=cv2.INTER_LINEAR)
                for e in edges
            ]
            edges = np.stack(edges, axis=2)
            edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
            if safe_steps != 0:
                edge = safe_step(edge, safe_steps)
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = edge

        detected_map = HWC3(detected_map)
        resize_result = fast_resize_image(
            detected_map, image_resolution
        )

        return Image.fromarray(resize_result)
