# =====================================
# Adetailer
# =====================================
from functools import partial
from diffusers import DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler
from asdff import yolo_detector
from huggingface_hub import hf_hub_download
from asdff.sd import AdCnPreloadPipe
import torch
import gc


def ad_model_process(
    adetailer,
    face_detector_ad,
    person_detector_ad,
    hand_detector_ad,
    # model_repo_id,
    # common,
    inpaint_only,
    image_list_task,  # pil
    mask_dilation=4,
    mask_blur=4,
    mask_padding=32,
):
    # input: params adetailer
    # output: list of PIL images

    adetailer.inpaint_pipeline.safety_checker = None
    adetailer.inpaint_pipeline.to("cuda")

    image_list_ad = []

    common = inpaint_only

    for img_single in image_list_task:
        images_ad = img_single.convert("RGB")

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

        result_ad = adetailer(
            images=[images_ad],
            common=common,
            inpaint_only=inpaint_only,  # {**inpaint_only, "strength": 0.4}
            detectors=detectors,
            mask_dilation=mask_dilation,
            mask_blur=mask_blur,
            mask_padding=mask_padding,
        )

        torch.cuda.empty_cache()
        gc.collect()

        image_list_ad.append(result_ad[0][0])

    torch.cuda.empty_cache()
    gc.collect()

    return image_list_ad
