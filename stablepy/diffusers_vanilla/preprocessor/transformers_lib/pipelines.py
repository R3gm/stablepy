from transformers import pipeline, AutoImageProcessor, SegformerForSemanticSegmentation, UperNetForSemanticSegmentation
import torch
import PIL
import numpy as np
from ..image_utils import HWC3, resize_image, ade_palette
import cv2


class ZoeDepth:
    def __init__(self):
        self.model = pipeline(task="depth-estimation", model="Intel/zoedepth-nyu-kitti", device=-1)

    @torch.inference_mode()
    def __call__(self, image: np.ndarray, **kwargs) -> PIL.Image.Image:
        detect_resolution = kwargs.pop("detect_resolution", 512)
        image_resolution = kwargs.pop("image_resolution", 512)
        image = HWC3(image)
        image = resize_image(image, resolution=detect_resolution)
        image = PIL.Image.fromarray(image)

        result = self.model(image)
        depth = result["depth"]

        depth_array = np.array(depth)
        depth_inverted = np.max(depth_array) - depth_array
        depth_inverted = HWC3(depth_inverted.astype(np.uint8))

        resize_result = resize_image(
            depth_inverted, resolution=image_resolution, interpolation=cv2.INTER_NEAREST
        )

        return PIL.Image.fromarray(resize_result)

    def to(self, device):
        self.model.device = torch.device(device)
        self.model.model.to(device)


class DPTDepthEstimator:
    def __init__(self):
        self.model = pipeline("depth-estimation", device=-1)

    def __call__(self, image: np.ndarray, **kwargs) -> PIL.Image.Image:
        detect_resolution = kwargs.pop("detect_resolution", 512)
        image_resolution = kwargs.pop("image_resolution", 512)
        image = np.array(image)
        image = HWC3(image)
        image = resize_image(image, resolution=detect_resolution)
        image = PIL.Image.fromarray(image)
        image = self.model(image)
        image = image["depth"]
        image = np.array(image)
        image = HWC3(image)
        image = resize_image(image, resolution=image_resolution)
        return PIL.Image.fromarray(image)

    def to(self, device):
        self.model.model.to(device)
        self.model.device = torch.device(device)


class UP_ImageSegmentor:
    def __init__(self):
        self.image_processor = AutoImageProcessor.from_pretrained(
            "openmmlab/upernet-convnext-small"
        )
        self.image_segmentor = UperNetForSemanticSegmentation.from_pretrained(
            "openmmlab/upernet-convnext-small"
        )

    @torch.inference_mode()
    def __call__(self, image: np.ndarray, **kwargs) -> PIL.Image.Image:
        detect_resolution = kwargs.pop("detect_resolution", 512)
        image_resolution = kwargs.pop("image_resolution", 512)
        image = HWC3(image)
        image = resize_image(image, resolution=detect_resolution)
        image = PIL.Image.fromarray(image)

        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values
        outputs = self.image_segmentor(pixel_values)
        seg = self.image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(ade_palette()):
            color_seg[seg == label, :] = color
        color_seg = color_seg.astype(np.uint8)

        color_seg = resize_image(
            color_seg, resolution=image_resolution, interpolation=cv2.INTER_NEAREST
        )
        return PIL.Image.fromarray(color_seg)


class SegFormer:
    def __init__(self):
        self.image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.image_segmentor = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

    @torch.inference_mode()
    def __call__(self, image: np.ndarray, **kwargs) -> PIL.Image.Image:
        detect_resolution = kwargs.pop("detect_resolution", 512)
        image_resolution = kwargs.pop("image_resolution", 512)
        image = HWC3(image)
        image = resize_image(image, resolution=detect_resolution)
        image = PIL.Image.fromarray(image)

        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values.to(self.image_segmentor.device.type)
        outputs = self.image_segmentor(pixel_values)
        seg = self.image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0].cpu()
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(ade_palette()):
            color_seg[seg == label, :] = color
        color_seg = color_seg.astype(np.uint8)

        color_seg = resize_image(
            color_seg, resolution=image_resolution, interpolation=cv2.INTER_NEAREST
        )
        return PIL.Image.fromarray(color_seg)

    def to(self, device):
        self.image_segmentor.to(device)


class DepthAnything:
    def __init__(self):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
        self.model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")

    @torch.inference_mode()
    def __call__(self, image: np.ndarray, **kwargs) -> PIL.Image.Image:
        detect_resolution = kwargs.pop("detect_resolution", 512)
        image_resolution = kwargs.pop("image_resolution", 512)
        image = HWC3(image)
        image = resize_image(image, resolution=detect_resolution)
        image = PIL.Image.fromarray(image)

        inputs = self.image_processor(images=image, return_tensors="pt").to(self.model.device.type)

        with torch.no_grad():
            outputs = self.model(**inputs)

        post_processed_output = self.image_processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.height, image.width)],
        )

        predicted_depth = post_processed_output[0]["predicted_depth"]
        depth = predicted_depth * 255 / predicted_depth.max()
        depth = depth.detach().cpu().numpy()

        depth = HWC3(depth.astype(np.uint8))
        resize_result = resize_image(
            depth, resolution=image_resolution, interpolation=cv2.INTER_NEAREST
        )

        return PIL.Image.fromarray(resize_result)

    def to(self, device):
        self.model.to(device)
