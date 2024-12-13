from PIL import Image
import numpy as np

from ..logging.logging_setup import logger
from ..diffusers_vanilla.utils import release_resources

FACE_RESTORATION_MODELS = ["CodeFormer", "GFPGAN", "RestoreFormer"]


def load_face_restoration_model(face_restoration_model, device):

    if face_restoration_model == FACE_RESTORATION_MODELS[0]:
        from .codeformer import FaceRestorerCodeFormer
        model = FaceRestorerCodeFormer(device)
    elif face_restoration_model == FACE_RESTORATION_MODELS[1]:
        from .gfpgan import FaceRestorerGFPGAN
        model = FaceRestorerGFPGAN(device)
    elif face_restoration_model == FACE_RESTORATION_MODELS[2]:
        from .restoreformer import FaceRestorerRestoreFormer
        model = FaceRestorerRestoreFormer(device)
    else:
        logger.error("Invalid face restoration model")
        return None
    logger.info(f"Face restoration: {face_restoration_model}")
    model.load_net()
    return model


def process_face_restoration(
    source_img, model, face_restoration_visibility, face_restoration_weight
):
    if face_restoration_visibility == 0 or model is None:
        return source_img

    source_img = source_img.convert("RGB")

    restored_img = model.restore(
        np.array(source_img, dtype=np.uint8), w=face_restoration_weight
    )
    res = Image.fromarray(restored_img)

    if face_restoration_visibility < 1.0:
        res = Image.blend(source_img, res, face_restoration_visibility)

    return res


def batch_process_face_restoration(
    images,
    face_restoration_model,
    face_restoration_visibility,
    face_restoration_weight,
    device="cuda",
):

    model = load_face_restoration_model(face_restoration_model, device)

    result_list = []
    for source_img in images:
        try:
            res = process_face_restoration(
                source_img,
                model,
                face_restoration_visibility,
                face_restoration_weight
            )
            result_list.append(res)
        except Exception:
            logger.error("Failed face restoration", exc_info=True)
            result_list.append(source_img)

    del model
    release_resources()

    return result_list
