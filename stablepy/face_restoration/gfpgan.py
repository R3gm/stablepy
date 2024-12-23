from __future__ import annotations

import torch

from ..upscalers.utils_upscaler import load_spandrel_model, load_file_from_url
from .face_restoration_utils import CommonFaceRestoration

model_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
model_download_name = "GFPGANv1.4.pth"


class FaceRestorerGFPGAN(CommonFaceRestoration):
    def name(self):
        return "GFPGAN"

    def get_device(self):
        return self.device

    def load_net(self, path=None) -> torch.Module:
        if not path:
            path = model_url

        if path.startswith("http"):
            path = load_file_from_url(
                path,
                model_dir=self.model_path,
            )

        return load_spandrel_model(
            path,
            device=self.get_device(),
            expected_architecture='GFPGAN',
            prefer_half=False,
        ).model

    def restore(self, np_image, **kwargs):
        def restore_face(cropped_face_t):
            assert self.net is not None
            return self.net(cropped_face_t, return_rgb=False)[0]

        return self.restore_with_helper(np_image, restore_face)
