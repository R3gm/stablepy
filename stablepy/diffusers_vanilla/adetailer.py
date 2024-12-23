# =====================================
# Adetailer
# =====================================
from functools import partial
from diffusers import DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, DDIMScheduler
from huggingface_hub import hf_hub_download
from typing import Any, Callable, Iterable, List, Mapping, Optional
import numpy as np
from PIL import Image
import torch, copy, gc
from ..logging.logging_setup import logger

FIXED_SIZE_CLASS = [
    "StableDiffusionControlNetInpaintPipeline",
    "StableDiffusionXLInpaintPipeline",
    "FluxImg2ImgPipeline",
    "FluxInpaintPipeline",
]


def ad_model_process(
    detailfix_pipe,
    pipe_params_df,
    face_detector_ad,
    person_detector_ad,
    hand_detector_ad,
    image_list_task,  # pil
    mask_dilation=4,
    mask_blur=4,
    mask_padding=32,
):
    # input: params pipe, detailfix_pipe, paras yolo
    # output: list of PIL images

    scheduler_assigned = copy.deepcopy(detailfix_pipe.scheduler)
    logger.debug(f"Base sampler detailfix_pipe: {scheduler_assigned}")

    detailfix_pipe.safety_checker = None
    detailfix_pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    # detailfi resolution param
    if str(detailfix_pipe.__class__.__name__) in FIXED_SIZE_CLASS:
        pipe_params_df["height"] = image_list_task[0].size[1]
        pipe_params_df["width"] = image_list_task[0].size[0]
        logger.debug(f"detailfix inpaint only")
    else:
        pipe_params_df.pop("height", None)
        pipe_params_df.pop("width", None)
        logger.debug(f"detailfix img2img")

    image_list_ad = []

    detectors = []
    if person_detector_ad:
        person_model_path = hf_hub_download("Bingsu/adetailer", "person_yolov8s-seg.pt")
        person_detector = partial(yolo_detector, model_path=person_model_path)
        detectors.append(person_detector)
    if face_detector_ad:
        face_model_path = hf_hub_download("Bingsu/adetailer", "face_yolov8n.pt")
        face_detector = partial(yolo_detector, model_path=face_model_path)
        detectors.append(face_detector)
    if hand_detector_ad:
        hand_model_path = hf_hub_download("Bingsu/adetailer", "hand_yolov8n.pt")
        hand_detector = partial(yolo_detector, model_path=hand_model_path)
        detectors.append(hand_detector)

    image_list_ad = []

    for i, init_image_base in enumerate(image_list_task):
        init_image = init_image_base.convert("RGB")
        final_image = None

        for j, detector in enumerate(detectors):
            masks = detector(init_image)

            if masks is None:
                logger.info(
                    f"No object detected on {(i + 1)} image with {str(detector).split('/')[-1][:-2]} detector."
                )
                continue

            for k, mask in enumerate(masks):
                mask = mask.convert("L")
                mask = mask_dilate(mask, mask_dilation)
                bbox = mask.getbbox()
                if bbox is None:
                    logger.info(f"No object in {(k + 1)} mask.")
                    continue
                mask = mask_gaussian_blur(mask, mask_blur)
                bbox_padded = bbox_padding(bbox, init_image.size, mask_padding)

                crop_image = init_image.crop(bbox_padded)
                crop_mask = mask.crop(bbox_padded)

                pipe_params_df["image"] = crop_image
                pipe_params_df["mask_image"] = crop_mask

                if str(detailfix_pipe.__class__.__name__) == "StableDiffusionControlNetInpaintPipeline":
                    logger.debug("SD 1.5 detailfix")
                    pipe_params_df["control_image"] = make_inpaint_condition(crop_image, crop_mask)

                try:
                    inpaint_output = detailfix_pipe(**pipe_params_df)
                except Exception as e:
                    e = str(e)
                    if "Tensor with 2 elements cannot be converted to Scalar" in e:
                        try:
                            logger.error("Sampler not compatible with DetailFix; trying with DDIM sampler")
                            logger.debug(e)
                            detailfix_pipe.scheduler = detailfix_pipe.default_scheduler
                            detailfix_pipe.scheduler = DDIMScheduler.from_config(detailfix_pipe.scheduler.config)

                            inpaint_output = detailfix_pipe(**pipe_params_df)
                        except Exception as ex:
                            logger.error("trying with base sampler")
                            logger.debug(str(ex))
                            detailfix_pipe.scheduler = detailfix_pipe.default_scheduler

                            inpaint_output = detailfix_pipe(**pipe_params_df)
                    elif "The size of tensor a (0) must match the size of tensor b (3) at non-singleton" in e or "cannot reshape tensor of 0 elements into shape [0, -1, 1, 512] because the unspecified dimensi" in e:
                        logger.error(f"strength or steps too low for the model to produce a satisfactory response.")
                        inpaint_output = [[crop_image]]
                    else:
                        raise ValueError(e)

                inpaint_image: Image.Image = inpaint_output[0][0]
                final_image = composite(
                    init=init_image,
                    mask=mask,
                    gen=inpaint_image,
                    bbox_padded=bbox_padded,
                )
                init_image = final_image

        if final_image is not None:
            image_list_ad.append(final_image)
        else:
            logger.info(
                f"DetailFix: No detections found in image. Returning original image"
            )
            image_list_ad.append(init_image_base)

        torch.cuda.empty_cache()
        gc.collect()

    detailfix_pipe.scheduler = scheduler_assigned

    torch.cuda.empty_cache()
    gc.collect()

    return image_list_ad


# =====================================
# Yolo
# =====================================
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image
from ultralytics import YOLO


def create_mask_from_bbox(
    bboxes: np.ndarray, shape: tuple[int, int]
) -> list[Image.Image]:
    """
    Parameters
    ----------
        bboxes: list[list[float]]
            list of [x1, y1, x2, y2]
            bounding boxes
        shape: tuple[int, int]
            shape of the image (width, height)

    Returns
    -------
        masks: list[Image.Image]
        A list of masks

    """
    masks = []
    for bbox in bboxes:
        mask = Image.new("L", shape, "black")
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(bbox, fill="white")
        masks.append(mask)
    return masks


def mask_to_pil(masks: torch.Tensor, shape: tuple[int, int]) -> list[Image.Image]:
    """
    Parameters
    ----------
    masks: torch.Tensor, dtype=torch.float32, shape=(N, H, W).
        The device can be CUDA, but `to_pil_image` takes care of that.

    shape: tuple[int, int]
        (width, height) of the original image

    Returns
    -------
    images: list[Image.Image]
    """
    n = masks.shape[0]
    return [to_pil_image(masks[i], mode="L").resize(shape) for i in range(n)]


def yolo_detector(
    image: Image.Image, model_path: str | Path | None = None, confidence: float = 0.3
) -> list[Image.Image] | None:
    if not model_path:
        model_path = hf_hub_download("Bingsu/adetailer", "face_yolov8n.pt")
    model = YOLO(model_path)
    pred = model(image, conf=confidence)

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    if bboxes.size == 0:
        return None

    if pred[0].masks is None:
        masks = create_mask_from_bbox(bboxes, image.size)
    else:
        masks = mask_to_pil(pred[0].masks.data, image.size)

    return masks


# =====================================
# Utils
# =====================================

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torch


def mask_dilate(image: Image.Image, value: int = 4) -> Image.Image:
    if value <= 0:
        return image

    arr = np.array(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))
    dilated = cv2.dilate(arr, kernel, iterations=1)
    return Image.fromarray(dilated)


def mask_gaussian_blur(image: Image.Image, value: int = 4) -> Image.Image:
    if value <= 0:
        return image

    blur = ImageFilter.GaussianBlur(value)
    return image.filter(blur)


def bbox_padding(
    bbox: tuple[int, int, int, int], image_size: tuple[int, int], value: int = 32
) -> tuple[int, int, int, int]:
    if value <= 0:
        return bbox

    arr = np.array(bbox).reshape(2, 2)
    arr[0] -= value
    arr[1] += value
    arr = np.clip(arr, (0, 0), image_size)
    return tuple(arr.flatten())


def composite(
    init: Image.Image,
    mask: Image.Image,
    gen: Image.Image,
    bbox_padded: tuple[int, int, int, int],
) -> Image.Image:
    img_masked = Image.new("RGBa", init.size)
    img_masked.paste(
        init.convert("RGBA").convert("RGBa"),
        mask=ImageOps.invert(mask),
    )
    img_masked = img_masked.convert("RGBA")

    size = (
        bbox_padded[2] - bbox_padded[0],
        bbox_padded[3] - bbox_padded[1],
    )
    resized = gen.resize(size)

    output = Image.new("RGBA", init.size)
    output.paste(resized, bbox_padded)
    output.alpha_composite(img_masked)
    return output.convert("RGB")



def make_inpaint_condition(init_image, mask_image):
    init_image = np.array(init_image.convert("RGB")).astype(np.float32) / 255.0
    mask_image = np.array(mask_image.convert("L")).astype(np.float32) / 255.0

    assert init_image.shape[0:1] == mask_image.shape[0:1], "image and image_mask must have the same image size"
    init_image[mask_image > 0.5] = -1.0  # set as masked pixel
    init_image = np.expand_dims(init_image, 0).transpose(0, 3, 1, 2)
    init_image = torch.from_numpy(init_image)
    return init_image
