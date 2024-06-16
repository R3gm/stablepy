# stablepy

**Description:**

The goal of this project is to make Stable Diffusion more accessible, simple and easy-to-use with python.  Stablepy is constructed on top of the Diffusers library

**Installation:**
```
pip install stablepy==0.4.0
```

**Usage:**

To use the project, simply create a new instance of the `Model_Diffusers` class. This class takes several arguments, including the path to the Stable Diffusion model file and the task name.

Once you have created a new instance of the `Model_Diffusers` class, you can call the `model()` method to generate an image. The `model()` method takes several arguments, including the prompt, the number of steps, the guidance scale, the sampler, the image width, the image height, the path to the upscaler model (if using), etc.

**Interactive tutorial:**

See [stablepy_demo.ipynb](https://github.com/R3gm/stablepy/blob/main/stablepy_demo.ipynb)

<a target="_blank" href="https://colab.research.google.com/github/R3gm/stablepy/blob/main/stablepy_demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


**Examples:**

The following code examples show how to use the project to generate a text-to-image and a ControlNet diffusion:

```python
from stablepy import Model_Diffusers

# Generate a text-to-image diffusion
model = Model_Diffusers(
    base_model_id='./models/toonyou_beta6.safetensors',
    task_name= 'txt2img',
)

image, path_image = model(
    prompt='highly detailed portrait of an underwater city, with towering spires and domes rising up from the ocean floor',
    num_steps = 30,
    guidance_scale = 7.5,
    sampler = "DPM++ 2M",
    img_width = 512,
    img_height = 1024,
    upscaler_model_path = "./upscaler/RealESRGAN_x4plus_anime_6B.pth",
    upscaler_increases_size = 1.5,
    hires_steps = 25,
)

image[0]
```
Multiple LoRAs can also be used, as well as optimizations to the generation such as FreeU.
```python
from stablepy import Model_Diffusers

# Generate an ControlNet diffusion
model = Model_Diffusers(
    base_model_id='./models/toonyou_beta6.safetensors',
    task_name= 'canny',
)

images, path_images = model(
    prompt='highly detailed portrait of an underwater city, with towering spires and domes rising up from the ocean floor',
    num_steps = 30,
    image_resolution = 768,
    preprocessor_name = "Canny",
    guidance_scale = 7.5,
    seed = 567,
    FreeU = True,
    lora_A = "./loras/lora14552.safetensors",
    lora_scale_A = 0.8,
    lora_B = "./loras/example_lora3.safetensors",
    lora_scale_B = 0.5,
    image = "./examples/image001.png",
)

images[1]
```
**📖 News:**

🔥 Version 0.4.0: New Update Details

- IP Adapter with the variants FaceID and Instant-Style
- New samplers
- Appropriate support for SDXL safetensors models
- ControlNet for SDXL: OpenPose, Canny, Scribble, SoftEdge, Depth, LineArt, and SDXL_Tile_Realistic
- New variant prompt weight with emphasis
- ControlNet pattern for SD1.5 and SDXL
- ControlNet Canny now needs the `preprocessor_name="Canny"`
- Similarly, ControlNet MLSD requires the `preprocessor_name="MLSD"`
- Task names like "sdxl_canny" have been changed to "sdxl_canny_t2i" to refer to the T2I adapter that uses them.

**Contributing:**

We welcome contributions to the project. If you have any suggestions or bug fixes, please feel free to open an issue or submit a pull request.
