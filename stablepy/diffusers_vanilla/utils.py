# =====================================
# IMAGE: METADATA AND SAVE
# =====================================
import os
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from ..logging.logging_setup import logger

def save_pil_image_with_metadata(image, folder_path, metadata_list):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    existing_files = os.listdir(folder_path)

    # Determine the next available image name
    image_name = f"image{str(len(existing_files) + 1).zfill(3)}.png"
    image_path = os.path.join(folder_path, image_name)

    try:
        # metadata
        metadata = PngInfo()
        string_parameters = f"{str(metadata_list[0])}, Negative prompt: {str(metadata_list[1])} Steps: {str(metadata_list[4])}, Sampler: {str(metadata_list[6])}, CFG scale: {str(metadata_list[5])}, Seed: {str(metadata_list[7])}, Size: {str(metadata_list[8])}x{str(metadata_list[9])}, Model: {os.path.splitext(str(metadata_list[2]).split('/')[-1])[0]}, Clip skip: {2 if metadata_list[10] else 1},"
        metadata.add_text("parameters", string_parameters)
        # metadata.add_text("Prompt", str(metadata_list[0]))
        # metadata.add_text("Negative prompt", str(metadata_list[1]))
        # metadata.add_text("Model", str(metadata_list[2]))
        # metadata.add_text("VAE", str(metadata_list[3]))
        # metadata.add_text("Steps", str(metadata_list[4]))
        # metadata.add_text("CFG", str(metadata_list[5]))
        # metadata.add_text("Scheduler", str(metadata_list[6]))
        # metadata.add_text("Seed", str(metadata_list[7]))

        image.save(image_path, pnginfo=metadata)
    except:
        logger.info("Saving image without metadata")
        image.save(image_path)

    return image_path
