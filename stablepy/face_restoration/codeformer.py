from __future__ import annotations
import os

import torch

from ..upscalers.utils_upscaler import load_spandrel_model, load_file_from_url
from .face_restoration_utils import CommonFaceRestoration

model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'


class FaceRestorerCodeFormer(CommonFaceRestoration):
    def name(self):
        return "CodeFormer"

    def load_net(self, path=None) -> torch.Module:
        if not path:
            path = model_url

        if path.startswith("http"):
            path = load_file_from_url(
                path,
                model_dir=self.model_path,
            )

        if os.path.exists(path):
            return load_spandrel_model(
                path,
                device=self.device,
                expected_architecture='CodeFormer',
                prefer_half=False,
            ).model
        raise ValueError("No codeformer model found")

    def get_device(self):
        return self.device

    def restore(self, np_image, w: float | None = None, **kwargs):
        if w is None:
            w = 0.5

        def restore_face(cropped_face_t):
            assert self.net is not None
            return self.net(cropped_face_t, weight=w, adain=True)[0]

        return self.restore_with_helper(np_image, restore_face)
