
# =====================================
# IMAGE: METADATA AND SAVE
# =====================================
import os
from PIL import Image
from PIL.PngImagePlugin import PngInfo

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
        metadata.add_text("Prompt", str(metadata_list[0]))
        metadata.add_text("Negative prompt", str(metadata_list[1]))
        metadata.add_text("Model", str(metadata_list[2]))
        metadata.add_text("VAE", str(metadata_list[3]))
        metadata.add_text("Steps", str(metadata_list[4]))
        metadata.add_text("CFG", str(metadata_list[5]))
        metadata.add_text("Scheduler", str(metadata_list[6]))
        metadata.add_text("Seed", str(metadata_list[7]))

        image.save(image_path, pnginfo=metadata)
    except:
        print('Saving image without metadata')
        image.save(image_path)

    return image_path
