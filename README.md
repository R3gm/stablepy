# stablepy

**Description:**

The goal of this project is to make Stable Diffusion more accessible, simple and easy-to-use with python.  Stablepy is constructed on top of the Diffusers library

**Installation:**
```
pip install stablepy==0.6.1
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

image, info_image = model(
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
    # controlnet_model="r3gm/controlnet-union-promax-sdxl-fp16"  # controlnet_model by default is "Automatic"
)

images, info_image = model(
    prompt='highly detailed portrait of an underwater city, with towering spires and domes rising up from the ocean floor',
    num_steps = 30,
    image = "./examples/image001.png",
    image_resolution = 768, # Maximum resolution scaled while maintaining proportions based on original dimensions
    preprocessor_name = "Canny",
    guidance_scale = 7.5,
    seed = 567,
    FreeU = True,
    lora_A = "./loras/lora14552.safetensors",
    lora_scale_A = 0.8,
    lora_B = "./loras/example_lora3.safetensors",
    lora_scale_B = 0.5,
)

images[1]
```

**Parameters Model_Diffusers and Model_Diffusers.load_pipe:**

- `base_model_id` (str): The ID of the base model to load.
- `task_name` (str, optional, defaults to "txt2img"): The task name for the model.
- `vae_model` (str, optional, defaults to None): The VAE model to use. To use the BakedVAE in the model, use the same `base_model_id` here.
- `type_model_precision` (torch.dtype, optional, defaults to torch.float16): The precision type for the model.
- `reload` (bool, optional, defaults to False): Whether to reload the model even if it is already loaded.
- `retain_task_model_in_cache` (bool, optional, defaults to False): Whether to retain the task model in cache.
- `controlnet_model` (str, optional, defaults to "Automatic"): The controlnet model to use.

**Parameters Model_Diffusers.advanced_params:**

- `image_preprocessor_cuda_active` (bool, defaults to False): Enables CUDA for the image preprocessor when needed. This can help speed up the process for ControlNet tasks.

**Parameters generation:**

- `prompt` (str, optional): The prompt to guide image generation.
- `negative_prompt` (str, optional): The prompt to guide what to not include in image generation. Ignored when not using guidance (`guidance_scale < 1`).
- `img_height` (int, optional, defaults to 512): The height in pixels of the generated image.
- `img_width` (int, optional, defaults to 512): The width in pixels of the generated image.
- `num_images` (int, optional, defaults to 1): The number of images to generate per prompt.
- `num_steps` (int, optional, defaults to 30): The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
- `guidance_scale` (float, optional, defaults to 7.5): A higher guidance scale value encourages the model to generate images closely linked to the text `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
- `clip_skip` (bool, optional): Number of layers to be skipped from CLIP while computing the prompt embeddings. It can be placed on the penultimate (True) or last layer (False).
- `seed` (int, optional, defaults to -1): A seed for controlling the randomness of the image generation process. -1 designates a random seed.
- `sampler` (str, optional, defaults to "DPM++ 2M"): The sampler algorithm that defines how noise is gradually removed. To see all the valid sampler names, use the following code:
  ```python
  from stablepy import scheduler_names
  print(scheduler_names)
  ```
- `schedule_type` (str, optional, defaults to "Automatic"): The pattern controlling the rate at which noise is removed across each generation step. To see all the valid schedule_type names, use the following code:
  ```python
  from stablepy import SCHEDULE_TYPE_OPTIONS
  print(SCHEDULE_TYPE_OPTIONS)
  ```
- `schedule_prediction_type` (str, optional, defaults to "Automatic"): Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process), `sample` (directly predicts the noisy sample) or `v_prediction` (see section 2.4 of [Imagen Video](https://imagen.research.google/video/paper.pdf) paper). To see all the valid schedule_prediction_type names, use the following code:
  ```python
  from stablepy import SCHEDULE_PREDICTION_TYPE_OPTIONS
  print(SCHEDULE_PREDICTION_TYPE_OPTIONS)
  ```
- `syntax_weights` (str, optional, defaults to "Classic"): Specifies the type of syntax weights and emphasis used during generation. "Classic" is (word:weight), "Compel" is (word)weight. To see all the valid syntax weight options, use the following code:
  ```python
  from stablepy import ALL_PROMPT_WEIGHT_OPTIONS
  print(ALL_PROMPT_WEIGHT_OPTIONS)
  ```
- `lora_A` (str, optional): Placeholder for lora A parameter.
- `lora_scale_A` (float, optional, defaults to 1.0): Placeholder for lora scale A parameter.
- `lora_B` (str, optional): Placeholder for lora B parameter.
- `lora_scale_B` (float, optional, defaults to 1.0): Placeholder for lora scale B parameter.
- `lora_C` (str, optional): Placeholder for lora C parameter.
- `lora_scale_C` (float, optional, defaults to 1.0): Placeholder for lora scale C parameter.
- `lora_D` (str, optional): Placeholder for lora D parameter.
- `lora_scale_D` (float, optional, defaults to 1.0): Placeholder for lora scale D parameter.
- `lora_E` (str, optional): Placeholder for lora E parameter.
- `lora_scale_E` (float, optional, defaults to 1.0): Placeholder for lora scale E parameter.
- `lora_F` (str, optional): Placeholder for lora F parameter.
- `lora_scale_F` (float, optional, defaults to 1.0): Placeholder for lora scale F parameter.
- `lora_G` (str, optional): Placeholder for lora G parameter.
- `lora_scale_G` (float, optional, defaults to 1.0): Placeholder for lora scale G parameter.
- `textual_inversion` (List[Tuple[str, str]], optional, defaults to []): Placeholder for textual inversion list of tuples. Helps the model to adapt to a particular style. [("<token_activation>","<path_embedding>"),...]
- `FreeU` (bool, optional, defaults to False): Is a method that substantially improves diffusion model sample quality at no costs.
- `adetailer_A` (bool, optional, defaults to False): Guided Inpainting to Correct Image, it is preferable to use low values for strength.
- `adetailer_A_params` (Dict[str, Any], optional, defaults to {}): Placeholder for adetailer_A parameters in a dict example {"prompt": "my prompt", "inpaint_only": True ...}. If not specified, default values will be used:
  - `face_detector_ad` (bool): Indicates whether face detection is enabled. Defaults to True.
  - `person_detector_ad` (bool): Indicates whether person detection is enabled. Defaults to True.
  - `hand_detector_ad` (bool): Indicates whether hand detection is enabled. Defaults to False.
  - `prompt` (str): A prompt for the adetailer_A. Defaults to an empty string.
  - `negative_prompt` (str): A negative prompt for the adetailer_A. Defaults to an empty string.
  - `strength` (float): The strength parameter value. Defaults to 0.35.
  - `mask_dilation` (int): The mask dilation value. Defaults to 4.
  - `mask_blur` (int): The mask blur value. Defaults to 4.
  - `mask_padding` (int): The mask padding value. Defaults to 32.
  - `inpaint_only` (bool): Indicates if only inpainting is to be performed. Defaults to True. False is img2img mode
  - `sampler` (str): The sampler type to be used. Defaults to "Use same sampler".
- `adetailer_B` (bool, optional, defaults to False): Guided Inpainting to Correct Image, it is preferable to use low values for strength.
- `adetailer_B_params` (Dict[str, Any], optional, defaults to {}): Placeholder for adetailer_B parameters in a dict example {"prompt": "my prompt", "inpaint_only": True ...}. If not specified, default values will be used.
- `style_prompt` (str, optional): If a style that is in STYLE_NAMES is specified, it will be added to the original prompt and negative prompt.
- `style_json_file` (str, optional): JSON with styles to be applied and used in style_prompt.
- `pag_scale` (float, optional): Perturbed Attention Guidance (PAG) enhances image generation quality without the need for training. If it is used, it is recommended to use values close to 3.0 for good results.
- `upscaler_model_path` (str, optional): This is the path of the model that will be used for the upscale; on the other hand, you can also simply use any of the built-in upscalers like 'Nearest', 'Latent', 'SwinIR 4x', etc. that can be consulted in the following code:
  ```python
  from stablepy import ALL_BUILTIN_UPSCALERS
  print(ALL_BUILTIN_UPSCALERS)
  ```
- `upscaler_increases_size` (float, optional, defaults to 1.5): Placeholder for upscaler increases size parameter.
- `upscaler_tile_size` (int, optional, defaults to 192): Tile if use a upscaler model.
- `upscaler_tile_overlap` (int, optional, defaults to 8): Tile overlap if use a upscaler model.
- `hires_steps` (int, optional, defaults to 25): The number of denoising steps for hires. More denoising steps usually lead to a higher quality image at the expense of slower inference.
- `hires_denoising_strength` (float, optional, defaults to 0.35): Strength parameter for the hires.
- `hires_prompt` (str, optional): The prompt for hires. If not specified, the main prompt will be used.
- `hires_negative_prompt` (str, optional): The negative prompt for hires. If not specified, the main negative prompt will be used.
- `hires_sampler` (str, optional, defaults to "Use same sampler"): The sampler used for the hires generation process. If not specified, the main sampler will be used.
- `hires_schedule_type` (str, optional, defaults to "Use same schedule type"): The schedule type used for the hires generation process. If not specified, the main schedule will be used.
- `hires_guidance_scale` (float, optional, defaults to -1.): The guidance scale used for the hires generation process. If the value is set to -1. the main guidance_scale will be used.
- `face_restoration_model` (str, optional, default None): This is the name of the face restoration model that will be used. To see all the valid face restoration model names, use the following code:
  ```python
  from stablepy import FACE_RESTORATION_MODELS
  print(FACE_RESTORATION_MODELS)
  ```
- `face_restoration_visibility` (float, optional, defaults to 1.): The visibility of the restored face's changes.
- `face_restoration_weight` (float, optional, defaults to 1.): The weight value used for the CodeFormer model.
- `image` (Any, optional): The image to be used for the Inpaint, ControlNet, or T2I adapter.
- `image_resolution` (int, optional, defaults to 512): Image resolution for the Img2Img, Inpaint, ControlNet, or T2I adapter. This is the maximum resolution the image will be scaled to while maintaining its proportions, based on the original dimensions. For example, if you have a 512x512 image and set `image_resolution=1024`, it will be resized to 1024x1024 during image generation.
- `image_mask` (Any, optional): Path image mask for the Inpaint and Repaint tasks.
- `strength` (float, optional, defaults to 0.35): Strength parameter for the Inpaint, Repaint and Img2Img.
- `preprocessor_name` (str, optional, defaults to "None"): Preprocessor name for ControlNet tasks. To see the mapping of tasks with their corresponding preprocessor names, use the following code:
  ```python
  from stablepy import TASK_AND_PREPROCESSORS
  print(TASK_AND_PREPROCESSORS)
  ```
- `preprocess_resolution` (int, optional, defaults to 512): Preprocess resolution for the Inpaint, ControlNet, or T2I adapter. This is the resolution used by the preprocessor to preprocess the image. Lower resolution can work faster but provides fewer details, while higher resolution gives more detail but can be slower and requires more resources.
- `low_threshold` (int, optional, defaults to 100): Low threshold parameter for ControlNet and T2I Adapter Canny.
- `high_threshold` (int, optional, defaults to 200): High threshold parameter for ControlNet and T2I Adapter Canny.
- `value_threshold` (float, optional, defaults to 0.1): Value threshold parameter for ControlNet MLSD.
- `distance_threshold` (float, optional, defaults to 0.1): Distance threshold parameter for ControlNet MLSD.
- `recolor_gamma_correction` (float, optional, defaults to 1.0): Gamma correction parameter for ControlNet Recolor.
- `tile_blur_sigma` (int, optional, defaults to 9.0): Blur sigma parameter for ControlNet Tile.
- `controlnet_conditioning_scale` (float, optional, defaults to 1.0): The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added to the residual in the original `unet`. Used in ControlNet and Inpaint.
- `control_guidance_start` (float, optional, defaults to 0.0): The percentage of total steps at which the ControlNet starts applying. Used in ControlNet and Inpaint.
- `control_guidance_end` (float, optional, defaults to 1.0): The percentage of total steps at which the ControlNet stops applying. Used in ControlNet and Inpaint.
- `t2i_adapter_preprocessor` (bool, optional, defaults to True): Preprocessor for the image in sdxl_canny by default is True.
- `t2i_adapter_conditioning_scale` (float, optional, defaults to 1.0): The outputs of the adapter are multiplied by `t2i_adapter_conditioning_scale` before they are added to the residual in the original unet.
- `t2i_adapter_conditioning_factor` (float, optional, defaults to 1.0): The fraction of timesteps for which adapter should be applied. If `t2i_adapter_conditioning_factor` is `0.0`, adapter is not applied at all. If `t2i_adapter_conditioning_factor` is `1.0`, adapter is applied for all timesteps. If `t2i_adapter_conditioning_factor` is `0.5`, adapter is applied for half of the timesteps.
- `loop_generation` (int, optional, defaults to 1): The number of times the specified `num_images` will be generated.
- `display_images` (bool, optional, defaults to False): If you use a notebook, you will be able to display the images generated with this parameter.
- `image_display_scale` (float, optional, defaults to 1.): The proportional scale of the displayed image in the notebook.
- `save_generated_images` (bool, optional, defaults to True): By default, the generated images are saved in the current location within the 'images' folder. You can disable this with this parameter.
- `filename_pattern` (str, optional, defaults to "model,seed"): Sets the name that will be used to save the images. This name can be any text or a specific key, and each value needs to be separated by a comma. You can check the list of valid keys:
  ```python
  from stablepy import VALID_FILENAME_PATTERNS
  print(VALID_FILENAME_PATTERNS)
  ```
- `image_storage_location` (str, optional, defaults to "./images"): The directory where the generated images are saved.
- `generator_in_cpu` (bool, optional, defaults to False): The generator by default is specified on the GPU. To obtain more consistent results across various environments, it is preferable to use the generator on the CPU.
- `leave_progress_bar` (bool, optional, defaults to False): Leave the progress bar after generating the image.
- `disable_progress_bar` (bool, optional, defaults to False): Do not display the progress bar during image generation.
- `hires_before_adetailer` (bool, optional, defaults to False): Apply an upscale and high-resolution fix before adetailer.
- `hires_after_adetailer` (bool, optional, defaults to True): Apply an upscale and high-resolution fix after adetailer.
- `retain_compel_previous_load` (bool, optional, defaults to False): The previous compel remains preloaded in memory.
- `retain_detailfix_model_previous_load` (bool, optional, defaults to False): The previous adetailer model remains preloaded in memory.
- `retain_hires_model_previous_load` (bool, optional, defaults to False): The previous hires model remains preloaded in memory.
- `ip_adapter_image` (Optional[Any], optional, default=[]): Image path or list of image paths for ip adapter.
- `ip_adapter_mask` (Optional[Any], optional, default=[]): Mask image path or list of mask image paths for ip adapter.
- `ip_adapter_model` (Optional[Any], optional, default=[]): Adapter model name or list of adapter model names. To see all the valid model names for SD1.5:
  ```python
  from stablepy import IP_ADAPTERS_SD
  print(IP_ADAPTERS_SD)
  ```
  To see all the valid model names for SDXL:
  ```python
  from stablepy import IP_ADAPTERS_SDXL
  print(IP_ADAPTERS_SDXL)
  ```
- `ip_adapter_scale` (Optional[Any], optional, default=[1.0]): Scaling factor or list of scaling factors for the ip adapter models.
- `ip_adapter_mode` (Optional[Any], optional, default=['original']): Adapter mode or list of adapter modes. Possible values are 'original', 'style', 'layout', 'style+layout'.
- `image_previews` (bool, optional, defaults to False): Displaying the image denoising process with a generator.
- `xformers_memory_efficient_attention` (bool, optional, defaults to False): Improves generation time; currently disabled due to quality issues with LoRA.
- `gui_active` (bool, optional, defaults to False): Utility when used with a GUI, it changes the behavior especially by displaying confirmation messages or options.
- `**kwargs` (dict, optional): kwargs is used to pass additional parameters to the Diffusers pipeline. This allows for flexibility when specifying optional settings like guidance_rescale, eta, cross_attention_kwargs, and more.

**📖 New Update Details:**

🔥 **Version 0.6.1:**

- Default device CPU for DPT and ZoeDepth.


🔥 **Version 0.6.0:**

- Schedule Types: Karras, Exponential, Beta, SGM Uniform, Simple, Lambdas, AYS timesteps, and AYS 10 steps.
- New Samplers: SA Solver and FlowMatch variants.
- Adjustable prediction type to support v-prediction models.
- Custom filenames for saved images.
- Adjust the proportional `image_display_size` in notebooks.
- Integration with `sd_embed` for prompt weights.
- Option for specify the ControlNet model path or Diffusers repository for tasks.
- Hi-res fix now allows configuration of CFG and schedule type.
- Cache the last two results from text encoders and ControlNet preprocessors.
- New Preprocessors: ZoeDepth, SegFormer, Depth Anything, TEED, Anyline, and LineartStandard.
- Seeds now follow an incremental numbering pattern.
- Tile blur sigma for ControlNet Tile, allowing adjustment of Gaussian blur strength.
- Additional parameters can be passed to the Diffusers pipeline if are valid (e.g., `guidance_rescale`, `eta`, etc.).
- Refactored upscaler to use Spandrel and support built-in upscalers.
- Compatibility with various types of upscalers. Refer to the list of compatible architectures: [Spandrel - Image Super Resolution](https://github.com/chaiNNer-org/spandrel#single-image-super-resolution).
- Parameters `esrgan_tile` and `esrgan_tile_overlap` have been renamed to `upscaler_tile_size` and `upscaler_tile_overlap`, respectively.
- Face Restoration Models: CodeFormer, GFPGAN, and RestoreFormer.
- Options to control visibility and weight for face restoration.
- ControlNet repaint feature, useful for inpainting and outpainting tasks.
- Proper support for ControlNet Union Promax.
- retain_task_model_in_cache is False by default.
- New IP adapters for anime-based models.
- Support for using up to a maximum of 7 LoRAs simultaneously.
- Minimal support for `FLUX.1-dev` and `FLUX.1-schnell` models in Safetensors and Diffusers formats.
- Environment components for FLUX to prevent continuous reloading of common components.
- Prompt embeds variants for FLUX models.

🔥 Version 0.5.2:

- Fixed an issue where errors occurred if PAG layers weren't turned off.
- Improved the generation quality for SDXL when using the Classic-variant prompt weight.
- Fixed a problem where special characters weren't handled correctly with the Classic prompt weight.

🔥 Version 0.5.1:

- After generation, PIL images are now returned along with sublists containing the seeds, image paths, and metadata with the parameters used in generation. `[pil_image], [seed, image_path, metadata]`
- The use of `image_previews=True` has been improved, and now preview images can be obtained during the generation steps using a generator. For more details, refer to the Colab notebook.

🔥 Version 0.5.0:

- Fix LoRA SDXL compatibility.
- Latent upscaler and variants.
- Perturbed Attention Guidance (PAG) enhances image generation quality without the need for training.
- Multiple images for one FaceID adapter.
- ControlNet for SDXL: MLSD, Segmentation, Normalbae.
- ControlNet "lineart_anime" task accessible and able to load a model different from the "lineart" task.
- ControlNet Tile and Recolor for SD1.5 and SDXL ("tile" replaces the previous task called "sdxl_tile_realistic").

🔥 Version 0.4.0:

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
