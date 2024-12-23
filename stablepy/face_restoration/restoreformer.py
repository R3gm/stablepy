from __future__ import annotations
import os

import torch

from ..upscalers.utils_upscaler import load_spandrel_model, load_file_from_url
from .face_restoration_utils import CommonFaceRestoration

model_url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'


class FaceRestorerRestoreFormer(CommonFaceRestoration):
    def name(self):
        return "RestoreFormer"

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
                expected_architecture='RestoreFormer',
                prefer_half=False,
            ).model
        raise ValueError("No RestoreFormer model found")

    def get_device(self):
        return self.device

    def restore(self, np_image, **kwargs):

        def restore_face(cropped_face_t):
            assert self.net is not None
            return self.net(cropped_face_t)[0]

        return self.restore_with_helper(np_image, restore_face)
