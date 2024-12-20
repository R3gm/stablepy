from PIL import Image
import numpy as np

from ..logging.logging_setup import logger
from ..diffusers_vanilla.utils import release_resources

FACE_RESTORATION_MODELS = ["CodeFormer", "GFPGAN", "RestoreFormer"]


def load_face_restoration_model(face_restoration_model, device):
    """
    Load the specified face restoration model.
    Parameters:
    face_restoration_model (str): The name of the face restoration model to load.
                                  Must be one of the models listed in FACE_RESTORATION_MODELS.
    device (str): The device to load the model on (e.g., 'cpu' or 'cuda').
    Returns:
    model: An instance of the specified face restoration model, or None if an invalid model name is provided.
    """

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
        valid_models = ", ".join(FACE_RESTORATION_MODELS)
        logger.error(f"Invalid face restoration model: {face_restoration_model}. Valid models are: {valid_models}")
        return None
    logger.info(f"Face restoration: {face_restoration_model}")
    model.load_net()
    return model


def process_face_restoration(
    source_img,
    model,
    face_restoration_visibility,
    face_restoration_weight
):
    """
    Process a single image for face restoration.

    Parameters:
    source_img (PIL.Image): The source image to be processed.
    model: The face restoration model to use.
    face_restoration_visibility (float): The visibility of the restored face in the final image.
                                         Should be between 0 and 1.
    face_restoration_weight (float): The weight parameter for CodeFormer model.

    Returns:
    PIL.Image: The processed image with face restoration applied.
    """
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
    """
    Processes a batch of images for face restoration using the specified model and parameters.
    Args:
        images (list): List of images to be processed.
        face_restoration_model (str): The name or path of the face restoration model to be used.
        face_restoration_visibility (float): The visibility parameter for face restoration.
        face_restoration_weight (float): The weight parameter for CodeFormer model.
        device (str, optional): The device to run the model on, either 'cuda' or 'cpu'. Defaults to 'cuda'.
    Returns:
        list: A list of processed images. If an error occurs during processing, the original image is returned in the list.
    """

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
        except Exception as e:
            logger.error(f"Failed face restoration: {str(e)}", exc_info=True)
            result_list.append(source_img)

    del model
    release_resources()

    return result_list
