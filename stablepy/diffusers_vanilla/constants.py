from stablepy.diffusers_vanilla.extra_scheduler.scheduling_euler_discrete_variants import (
    EulerDiscreteSchedulerNegative,
    EulerDiscreteSchedulerMax,
)
from .extra_scheduler.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler
)
from .extra_scheduler.scheduling_flow_match_dpmsolver_multistep import (
    FlowMatchDPMSolverMultistepScheduler
)
from diffusers import (
    # DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2DiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    DEISMultistepScheduler,
    UniPCMultistepScheduler,
    LCMScheduler,
    PNDMScheduler,
    KDPM2AncestralDiscreteScheduler,
    DPMSolverSDEScheduler,
    EDMDPMSolverMultistepScheduler,
    DDPMScheduler,
    EDMEulerScheduler,
    TCDScheduler,
    SASolverScheduler,
    FlowMatchEulerDiscreteScheduler,
    # FlowMatchHeunDiscreteScheduler,
)
from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLAdapterPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionPAGPipeline,
    StableDiffusionControlNetPAGInpaintPipeline,
    StableDiffusionControlNetPAGPipeline,
    # StableDiffusionControlNetImg2ImgPAGPipeline,
    StableDiffusionXLPAGPipeline,
    StableDiffusionXLPAGInpaintPipeline,
    StableDiffusionXLControlNetPAGPipeline,
    # StableDiffusionXLAdapterPAGPipeline,
    # StableDiffusionXLControlNetPAGInpaintPipeline,
    StableDiffusionXLControlNetPAGImg2ImgPipeline,
)
from .extra_pipe.sdxl.pipeline_controlnet_union_inpaint_sd_xl import StableDiffusionXLControlNetUnionInpaintPipeline
from .extra_pipe.sdxl.pipeline_controlnet_union_sd_xl import StableDiffusionXLControlNetUnionPipeline

SD15 = "StableDiffusionPipeline"
SDXL = "StableDiffusionXLPipeline"
FLUX = "FluxPipeline"

CLASS_DIFFUSERS_TASK = {
    SD15: {
        "base": StableDiffusionPipeline,
        "inpaint": StableDiffusionControlNetInpaintPipeline,  # default cn
        "controlnet": StableDiffusionControlNetPipeline,
        "controlnet_img2img": StableDiffusionControlNetImg2ImgPipeline,
        "controlnet_inpaint": StableDiffusionControlNetInpaintPipeline,
    },
    SDXL: {
        "base": StableDiffusionXLPipeline,
        "inpaint": StableDiffusionXLInpaintPipeline,
        "controlnet": StableDiffusionXLControlNetPipeline,
        "adapter": StableDiffusionXLAdapterPipeline,
        "controlnet_img2img": StableDiffusionXLControlNetImg2ImgPipeline,
        "controlnet_inpaint": StableDiffusionXLControlNetInpaintPipeline,
        "controlnet_union+": StableDiffusionXLControlNetUnionPipeline,
        "controlnet_union+_inpaint": StableDiffusionXLControlNetUnionInpaintPipeline,
    },
}

CLASS_PAG_DIFFUSERS_TASK = {
    SD15: {
        "base": StableDiffusionPAGPipeline,
        "inpaint": StableDiffusionControlNetPAGInpaintPipeline,  # default cn
        "controlnet": StableDiffusionControlNetPAGPipeline,
        # "controlnet_img2img": StableDiffusionControlNetImg2ImgPAGPipeline,
        "controlnet_inpaint": StableDiffusionControlNetPAGInpaintPipeline,
    },
    SDXL: {
        "base": StableDiffusionXLPAGPipeline,
        "inpaint": StableDiffusionXLPAGInpaintPipeline,
        "controlnet": StableDiffusionXLControlNetPAGPipeline,
        # "adapter": StableDiffusionXLAdapterPAGPipeline,
        "controlnet_img2img": StableDiffusionXLControlNetPAGImg2ImgPipeline,
        # "controlnet_inpaint": StableDiffusionXLControlNetPAGInpaintPipeline,
        # "controlnet_union+": StableDiffusionXLControlNetUnionPAGPipeline,
        # "controlnet_union+_inpaint": StableDiffusionXLControlNetUnionPAGInpaintPipeline,
    },
}

CONTROLNET_MODEL_IDS = {
    "openpose": ["lllyasviel/control_v11p_sd15_openpose", "r3gm/controlnet-openpose-sdxl-1.0-fp16"],
    "canny": ["lllyasviel/control_v11p_sd15_canny", "r3gm/controlnet-canny-scribble-integrated-sdxl-v2-fp16"],
    "mlsd": ["lllyasviel/control_v11p_sd15_mlsd", "r3gm/controlnet-union-sdxl-1.0-fp16"],
    "scribble": ["lllyasviel/control_v11p_sd15_scribble", "r3gm/controlnet-canny-scribble-integrated-sdxl-v2-fp16"],
    "softedge": ["lllyasviel/control_v11p_sd15_softedge", "r3gm/controlnet-union-sdxl-1.0-fp16"],
    "segmentation": ["lllyasviel/control_v11p_sd15_seg", "r3gm/controlnet-union-sdxl-1.0-fp16"],
    "depth": ["lllyasviel/control_v11f1p_sd15_depth", "r3gm/controlnet-union-sdxl-1.0-fp16"],
    "normalbae": ["lllyasviel/control_v11p_sd15_normalbae", "r3gm/controlnet-union-sdxl-1.0-fp16"],
    "lineart": ["lllyasviel/control_v11p_sd15_lineart", "r3gm/controlnet-union-sdxl-1.0-fp16"],
    "lineart_anime": ["lllyasviel/control_v11p_sd15s2_lineart_anime", "r3gm/controlnet-lineart-anime-sdxl-fp16"],
    "shuffle": "lllyasviel/control_v11e_sd15_shuffle",
    "ip2p": "lllyasviel/control_v11e_sd15_ip2p",
    "inpaint": "lllyasviel/control_v11p_sd15_inpaint",
    "txt2img": "Nothinghere",
    "sdxl_canny_t2i": "TencentARC/t2i-adapter-canny-sdxl-1.0",
    "sdxl_sketch_t2i": "TencentARC/t2i-adapter-sketch-sdxl-1.0",
    "sdxl_lineart_t2i": "TencentARC/t2i-adapter-lineart-sdxl-1.0",
    "sdxl_depth-midas_t2i": "TencentARC/t2i-adapter-depth-midas-sdxl-1.0",
    "sdxl_openpose_t2i": "TencentARC/t2i-adapter-openpose-sdxl-1.0",
    "img2img": "Nothinghere",
    "pattern": ["monster-labs/control_v1p_sd15_qrcode_monster", "r3gm/control_v1p_sdxl_qrcode_monster_fp16"],
    "tile": ["lllyasviel/control_v11f1e_sd15_tile", "r3gm/controlnet-tile-sdxl-1.0-fp16"],  # "sdxl_tile_realistic": "Yakonrus/SDXL_Controlnet_Tile_Realistic_v2",
    "recolor": ["latentcat/control_v1p_sd15_brightness", "r3gm/controlnet-recolor-sdxl-fp16"],
    "repaint": ["lllyasviel/control_v11p_sd15_inpaint", "brad-twinkl/controlnet-union-sdxl-1.0-promax"]
    # "sdxl_depth-zoe_t2i": "TencentARC/t2i-adapter-depth-zoe-sdxl-1.0",
    # "sdxl_recolor_t2i": "TencentARC/t2i-adapter-recolor-sdxl-1.0",
}

FLUX_CN_UNION_MODES = {
    "openpose": 4,
    "canny": 0,
    # "mlsd": 7,
    "scribble": 0,
    "softedge": 0,
    "segmentation": 6,
    "depth": 2,
    # "normalbae": 7,
    "lineart": 0,
    "lineart_anime": 0,
    # "shuffle": 7,
    # "ip2p": "7",
    "tile": [1, 3, 6],
    "recolor": 5,
}

SDXL_CN_UNION_PROMAX_MODES = {
    "openpose": 0,
    "canny": 3,
    "mlsd": 3,
    "scribble": 2,
    "softedge": 2,
    "segmentation": 5,
    "depth": 1,
    "normalbae": 4,
    "lineart": 3,
    "lineart_anime": 3,
    # "shuffle": 7,
    # "ip2p": 7,
    "tile": 6,
    "recolor": 6,
    "repaint": 7,
}

VALID_TASKS = list(CONTROLNET_MODEL_IDS.keys())
SD15_TASKS = [x for x in VALID_TASKS if ("sdxl" not in x.lower())]
SDXL_TASKS = [
    y for y in VALID_TASKS
    if (
        "sdxl" in y.lower()
        or isinstance(CONTROLNET_MODEL_IDS[y], list)
        or y in ["txt2img", "img2img", "inpaint"]
    )
]

OLD_PROMPT_WEIGHT_OPTIONS = {
    "Compel": "Compel",
    "Classic": "Classic",
}

SD_EMBED = {
    "Classic-sd_embed": "sd_embed",
}

CLASSIC_VARIANT = {
    "Classic-original": "Original",
    "Classic-no_norm": "No norm",
    "Classic-ignore": "Ignore",
    "None": "None",
}

PROMPT_WEIGHT_OPTIONS = {
    **OLD_PROMPT_WEIGHT_OPTIONS,
    **SD_EMBED,
    **CLASSIC_VARIANT
}

ALL_PROMPT_WEIGHT_OPTIONS = list(PROMPT_WEIGHT_OPTIONS.keys())

# Sampler: DPM++ 2M, Schedule type: Exponential
SCHEDULER_CONFIG_MAP = {
    "DPM++ 2M": (DPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver++", "use_karras_sigmas": False}),
    "DPM++ 2M SDE": (DPMSolverMultistepScheduler, {"use_karras_sigmas": False, "algorithm_type": "sde-dpmsolver++"}),
    "DPM++ 2S": (DPMSolverSinglestepScheduler, {"algorithm_type": "dpmsolver++", "use_karras_sigmas": False}),
    "DPM++ 1S": (DPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver++", "solver_order": 1}),
    "DPM 3M": (DPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver", "final_sigmas_type": "sigma_min", "solver_order": 3}),
    "DPM++ 3M": (DPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver++", "solver_order": 3}),
    "DPM++ 3M SDE": (DPMSolverMultistepScheduler, {"solver_order": 3, "algorithm_type": "sde-dpmsolver++"}),
    "DPM++ SDE": (DPMSolverSDEScheduler, {"use_karras_sigmas": False}),
    "DPM2": (KDPM2DiscreteScheduler, {}),
    "DPM2 a": (KDPM2AncestralDiscreteScheduler, {}),
    "Euler": (EulerDiscreteScheduler, {}),
    "Euler a": (EulerAncestralDiscreteScheduler, {}),
    "Heun": (HeunDiscreteScheduler, {}),
    "LMS": (LMSDiscreteScheduler, {}),
    "DDIM": (DDIMScheduler, {}),
    "DEIS": (DEISMultistepScheduler, {}),
    "UniPC": (UniPCMultistepScheduler, {}),
    "PNDM": (PNDMScheduler, {}),
    "Euler EDM": (EDMEulerScheduler, {}),
    "DPM++ 2M EDM": (EDMDPMSolverMultistepScheduler, {"solver_order": 2, "solver_type": "midpoint", "final_sigmas_type": "zero", "algorithm_type": "dpmsolver++"}),
    "DDPM": (DDPMScheduler, {}),
    "SA Solver": (SASolverScheduler, {"use_karras_sigmas": False, "timestep_spacing": "linspace"}),
    # "DPM++ 2M Lu": (DPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver++", "use_lu_lambdas": True}),
    "DPM++ 2M Ef": (DPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver++", "euler_at_final": True}),
    # "DPM++ 2M SDE Lu": (DPMSolverMultistepScheduler, {"use_lu_lambdas": True, "algorithm_type": "sde-dpmsolver++"}),
    "DPM++ 2M SDE Ef": (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++", "euler_at_final": True}),
    "DPM 3M Ef": (DPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver", "final_sigmas_type": "sigma_min", "solver_order": 3, "euler_at_final": True}),
    # "Euler Negative": (EulerDiscreteSchedulerNegative, {}),
    # "Euler Max": (EulerDiscreteSchedulerMax, {}),

    "LCM": (LCMScheduler, {}),
    "TCD": (TCDScheduler, {}),
    "LCM Auto-Loader": (LCMScheduler, {}),
    "TCD Auto-Loader": (TCDScheduler, {}),

    "FlowMatch Euler": (FlowMatchEulerDiscreteScheduler, {}),
    # "FlowMatch Heun": (FlowMatchHeunDiscreteScheduler, {}),
    "FlowMatch DPM2": (FlowMatchDPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver2", "eta": 1.0, "s_noise": 1.0, "use_noise_sampler": True}),
    "FlowMatch DPM++ 2M": (FlowMatchDPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver++2M", "eta": 1.0, "s_noise": 1.0, "use_noise_sampler": True}),
    "FlowMatch DPM++ 2S": (FlowMatchDPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver++2S", "eta": 1.0, "s_noise": 1.0, "use_noise_sampler": True}),
    "FlowMatch DPM++ SDE": (FlowMatchDPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver++sde", "eta": 1.0, "s_noise": 1.0, "use_noise_sampler": True}),
    "FlowMatch DPM++ 2M SDE": (FlowMatchDPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver++2Msde", "eta": 1.0, "s_noise": 1.0, "use_noise_sampler": True}),
    "FlowMatch DPM++ 3M SDE": (FlowMatchDPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver++3Msde", "eta": 1.0, "s_noise": 1.0, "use_noise_sampler": True}),
}

scheduler_names = list(SCHEDULER_CONFIG_MAP.keys())

FLASH_AUTO_LOAD_SAMPLER = ["LCM Auto-Loader", "TCD Auto-Loader"]

FLASH_LORA = {
    SD15: {
        FLASH_AUTO_LOAD_SAMPLER[0]: "latent-consistency/lcm-lora-sdv1-5",
        FLASH_AUTO_LOAD_SAMPLER[1]: "h1t/TCD-SD15-LoRA",
    },
    SDXL: {
        FLASH_AUTO_LOAD_SAMPLER[0]: "latent-consistency/lcm-lora-sdxl",
        FLASH_AUTO_LOAD_SAMPLER[1]: "h1t/TCD-SDXL-LoRA",
    },
}


AYS_SCHEDULES = {
    "AYS timesteps": [[999, 850, 736, 645, 545, 455, 343, 233, 124, 24], [999, 845, 730, 587, 443, 310, 193, 116, 53, 13]],
    "AYS 10 steps": [[999, 850, 736, 645, 545, 455, 343, 233, 124, 24], [999, 845, 730, 587, 443, 310, 193, 116, 53, 13]],  # [sd1.5, sdxl]
    "AYS sigmas": [[14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029], [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029]],  # [sd1.5, sdxl] # sampler Euler
    "AYS sigmas 10 steps": [[14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.0], [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.0]],  # [sd1.5, sdxl] # sampler Euler
}

SCHEDULE_TYPES = {
    "Automatic": "",
    "Karras": {"use_karras_sigmas": True},
    "Exponential": {"timestep_spacing": "linspace", "use_exponential_sigmas": True},
    "Beta": {"timestep_spacing": "linspace", "use_beta_sigmas": True},
    "SGM Uniform": {"timestep_spacing": "trailing"},
    "Normal": {"use_karras_sigmas": False},  # check
    "Simple": {"timestep_spacing": "trailing", "use_karras_sigmas": False},
    "Lambdas": {"use_lu_lambdas": True},
    "AYS timesteps": {"use_karras_sigmas": False},
    "AYS 10 steps": {"use_karras_sigmas": False},
    # "AYS sigmas": {"use_karras_sigmas": False},  # Euler
    # "AYS sigmas 10 steps": {"use_karras_sigmas": False},  # Euler
}

SCHEDULE_TYPE_OPTIONS = list(SCHEDULE_TYPES.keys())

SCHEDULE_PREDICTION_TYPE = {
    "Automatic": "",
    "Epsilon": {"prediction_type": "epsilon"},
    "Sample": {"prediction_type": "sample"},
    "V prediction": {"prediction_type": "v_prediction", "rescale_betas_zero_snr": True},
}

SCHEDULE_PREDICTION_TYPE_OPTIONS = list(SCHEDULE_PREDICTION_TYPE.keys())

FLUX_SCHEDULE_TYPES = {
    "Automatic": "",
    "Karras": {"sigma_schedule": "karras"},
    "Exponential": {"sigma_schedule": "exponential"},
    "Beta": {"sigma_schedule": "lambdas"},  # change key
}

FLUX_SCHEDULE_TYPE_OPTIONS = list(FLUX_SCHEDULE_TYPES.keys())

# Mixes that need fixing.
NOISE_IMAGE_STATUS = {}  # EDM

BLACK_IMAGE_STATUS = {
    "DPM 3M": ["Karras", "Exponential", "Beta", "Lambdas"],  #
}

ERROR_IMAGE_STATUS = {
    "DPM++ 2S": ["Exponential", "Beta"],
    "DEIS": ["Exponential", "Beta", "AYS timesteps", "AYS 10 steps"],
    "UniPC": ["Exponential", "Beta", "AYS timesteps", "AYS 10 steps"],
    "SA Solver": ["Exponential", "Beta", "AYS timesteps", "AYS 10 steps"],
    "DPM++ SDE": ["AYS timesteps", "AYS 10 steps"],
    "DPM2": ["AYS timesteps", "AYS 10 steps"],
    "DPM2 a": ["AYS timesteps", "AYS 10 steps"],
    "Euler a": ["AYS timesteps", "AYS 10 steps"],
    "LMS": ["AYS timesteps", "AYS 10 steps"],
    "DDIM": ["AYS timesteps", "AYS 10 steps"],
    "PNDM": ["AYS timesteps", "AYS 10 steps"],
    "Euler EDM": ["AYS timesteps", "AYS 10 steps"],
    "DPM++ 2M EDM": ["AYS timesteps", "AYS 10 steps"],
    # "DPM++ 2M Lu": ["AYS timesteps", "AYS 10 steps"],
    # "DPM++ 2M SDE Lu": ["AYS timesteps", "AYS 10 steps"],
}

INCOMPATIBILITY_SAMPLER_SCHEDULE = {}


# Function to merge each dictionary into the global dictionary
def merge_dicts(source_dict):
    for key, value in source_dict.items():
        if key not in INCOMPATIBILITY_SAMPLER_SCHEDULE:
            INCOMPATIBILITY_SAMPLER_SCHEDULE[key] = []
        INCOMPATIBILITY_SAMPLER_SCHEDULE[key].extend(value)


# Merge each dictionary
merge_dicts(NOISE_IMAGE_STATUS)
merge_dicts(BLACK_IMAGE_STATUS)
merge_dicts(ERROR_IMAGE_STATUS)

IP_ADAPTER_MODELS = {
    SD15: {
        # "img_encoder": ["h94/IP-Adapter", "models/image_encoder"],
        "full_face": ["h94/IP-Adapter", "models", "ip-adapter-full-face_sd15.safetensors", "H"],
        "plus_face": ["h94/IP-Adapter", "models", "ip-adapter-plus-face_sd15.safetensors", "H"],
        "plus": ["h94/IP-Adapter", "models", "ip-adapter-plus_sd15.safetensors", "H"],
        "base": ["h94/IP-Adapter", "models", "ip-adapter_sd15.safetensors", "H"],
        "base_vit_G": ["h94/IP-Adapter", "models", "ip-adapter_sd15_vit-G.safetensors", "G"],
        "base_light": ["h94/IP-Adapter", "models", "ip-adapter_sd15_light.safetensors", "H"],
        "base_light_v2": ["h94/IP-Adapter", "models", "ip-adapter_sd15_light_v11.bin", "H"],
        "faceid_plus": ["h94/IP-Adapter-FaceID", "", "ip-adapter-faceid-plus_sd15.bin", "H"],
        "faceid_plus_v2": ["h94/IP-Adapter-FaceID", "", "ip-adapter-faceid-plusv2_sd15.bin", "H"],
        "faceid": ["h94/IP-Adapter-FaceID", "", "ip-adapter-faceid_sd15.bin", None],
        "faceid_portrait_v2": ["h94/IP-Adapter-FaceID", "", "ip-adapter-faceid-portrait-v11_sd15.bin", None],
        "faceid_portrait": ["h94/IP-Adapter-FaceID", "", "ip-adapter-faceid-portrait_sd15.bin", None],
        "composition_plus": ["ostris/ip-composition-adapter", "", "ip_plus_composition_sd15.safetensors", "H"],
        "ipa_anime": ["r3gm/ip-adapter-anime", "", "ipAdapterAnimeFine_v10.safetensors", "H"],
        "ipa_anime_plus": ["r3gm/ip-adapter-anime", "", "ipAdapterPlusAnime_v10.safetensors", "H"],
    },
    SDXL: {
        # "img_encoder": ["h94/IP-Adapter", "sdxl_models/image_encoder"],
        "plus_face": ["h94/IP-Adapter", "sdxl_models", "ip-adapter-plus-face_sdxl_vit-h.safetensors", "H"],
        "plus": ["h94/IP-Adapter", "sdxl_models", "ip-adapter-plus_sdxl_vit-h.safetensors", "H"],
        "base_vit_G": ["h94/IP-Adapter", "sdxl_models", "ip-adapter_sdxl.safetensors", "G"],
        "base": ["h94/IP-Adapter", "sdxl_models", "ip-adapter_sdxl_vit-h.safetensors", "H"],
        "faceid_plus_v2": ["h94/IP-Adapter-FaceID", "", "ip-adapter-faceid-plusv2_sdxl.bin", "H"],
        "faceid": ["h94/IP-Adapter-FaceID", "", "ip-adapter-faceid_sdxl.bin", None],
        "faceid_portrait": ["h94/IP-Adapter-FaceID", "", "ip-adapter-faceid-portrait_sdxl.bin", None],
        "faceid_portrait_v2": ["h94/IP-Adapter-FaceID", "", "ip-adapter-faceid-portrait_sdxl_unnorm.bin", None],
        "composition_plus": ["ostris/ip-composition-adapter", "", "ip_plus_composition_sdxl.safetensors", "H"],
        "noob-ipa": ["r3gm/noob-ipa", "model_H", "pytorch_model.bin", "H"],
        "noob-ipa_vit_G": ["r3gm/noob-ipa", "model_G", "noobIPAMARK1_mark1.safetensors", "G"],
    }
}  # no suffix lora


def name_list_ip_adapters(model_key):
    adapters = list(IP_ADAPTER_MODELS[model_key].keys())
    if "img_encoder" in adapters:
        adapters.remove("img_encoder")
    return adapters


IP_ADAPTERS_SD = name_list_ip_adapters(SD15)
IP_ADAPTERS_SDXL = name_list_ip_adapters(SDXL)

REPO_IMAGE_ENCODER = {
    "H": ["h94/IP-Adapter", "models/image_encoder"],
    "G": ["h94/IP-Adapter", "sdxl_models/image_encoder"],
}

VALID_FILENAME_PATTERNS = [
    "prompt_section",
    "neg_prompt_section",
    "model",
    "vae",
    "num_steps",
    "guidance_scale",
    "sampler",
    "schedule_type",
    "img_width",
    "img_height",
    "seed",
]

BETA_STYLE_LIST = [
    {
        "name": "(No style)",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast",
    },
    {
        "name": "Manga",
        "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
        "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
        "negative_prompt": "photo, photorealistic, realism, ugly",
    },
    {
        "name": "Pixel art",
        "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
        "negative_prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
        "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white",
    },
    {
        "name": "Neonpunk",
        "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
        "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting",
    },
    {
        "name": "Steampunk Portrait",
        "prompt": "Steampunk inventor self-portrait {prompt}. Dramatic goggles, brass decor, detailed mechanical apparatus",
        "negative_prompt": "(Imperfections, asymmetric, messy arrangement, improper lighting)"
    },
    {
        "name": "Scary Stories",
        "prompt": "Spooky campfire stories {prompt}. Flickering flames, looming shadows, uneasy expressions, night woods",
        "negative_prompt": "(Joyful expressions, silly ghosts, rainbow colors, unicorns)"
    },
    {
        "name": "Woodblock Art",
        "prompt": "Japanese Edo period woodblock print {prompt}. Ukiyo-e style, inky textures, tranquil nature scene",
        "negative_prompt": "(Full bleed colors, empty background, shapes only)"
    },
    {
        "name": "Vintage Halloween Mask",
        "prompt": "Creepy vintage Halloween mask {prompt}. Distressed burlap, stitched seams, eerie expression",
        "negative_prompt": "(Hyper-realistic rendering, futuristic alien design, clean and new)"
    },
    {
        "name": "Grunge Flyer",
        "prompt": "Grunge music concert flyer {prompt}. Ripped paper, angular font, punk graphics, tape, staples",
        "negative_prompt": "(Conservative palette, delicate paper, intricate details, busy ornate)"
    },
    {
        "name": "Voodoo Altar",
        "prompt": "Elaborate voodoo altar {prompt}. Occult artifacts, candles, incense, offerings, dramatic lighting",
        "negative_prompt": "(Corporate office setting, desk and computer)"
    },
    {
        "name": "Cassette Collage",
        "prompt": "Surreal vaporwave cassette collage {prompt}. Glitch effects, nostalgic textures, retro sci-fi pop art",
        "negative_prompt": "(Cohesive unified art style, clean edges, strong focal point)"
    },
    {
        "name": "Tropical Cocktail",
        "prompt": "Fruity tropical cocktail {prompt}. Orchid, pineapple wedge, tiki mug, luau vibes, island sunset",
        "negative_prompt": "(Wilted garnish, cloudy liquid, bland common glass)"
    },
    {
        "name": "Vintage Robot Toy",
        "prompt": "Retro vintage robot toy {prompt}. Colorful plastic, lights and buttons, 1950s sci-fi influences",
        "negative_prompt": "(Realistic human scale, gritty worn metals, in action scene)"
    },
    {
        "name": "Scary Pumpkin",
        "prompt": "Creepy carved Halloween pumpkin {prompt}. Flickering candle, sinister grin, dramatic shadows",
        "negative_prompt": "(Cute face, soft lighting, happy expression, placed neatly)"
    },
    {
        "name": "Pinup",
        "prompt": "Vintage pinup style {prompt}. Retro swimsuits, playful poses, vibrant backdrop, kitschy Americana",
        "negative_prompt": "(Imperfections accentuated, wearing winter jacket and beanie)"
    },
    {
        "name": "Tarot Cards",
        "prompt": "Mysterious trio of tarot cards {prompt}. Dramatic lighting, occult symbols, divination ritual",
        "negative_prompt": "(Photo of ordinary playing cards, poker chips)"
    },
    {
        "name": "Ghibli",
        "prompt": "Studio Ghibli style {prompt}. Dreamlike wonder, hand painted backgrounds, youthful characters, wholesome",
        "negative_prompt": "(Violent, scary, hyperrealistic, dystopian, photograph)"
    },
    {
        "name": "Egyptology",
        "prompt": "Ancient Egyptian style {prompt}. Hieroglyphs, statues, sarcophagi, archaeology themes, earthy tones",
        "negative_prompt": "(Futuristic, digital, computer generated, plastic, neon, cluttered)"
    },
    {
        "name": "Battle",
        "prompt": "Epic battle scene {prompt}. Dramatic poses, swords and armor, ancient warriors clashing, gritty realism",
        "negative_prompt": "(Nonviolent imagery, neutral poses, pleasant expressions, clean, sparse)"
    },
    {
        "name": "Tarot",
        "prompt": "Mystical tarot card {prompt}. Elaborate borders, occult symbols, dramatic lighting, divination themes",
        "negative_prompt": "(Corporate, business themes, scientific motifs, clear lighting, minimalist)"
    },
    {
        "name": "Steampunk",
        "prompt": "Steampunk style {prompt}. Ornate brass decor, gears, goggles, Victorian fashion, science fantasy",
        "negative_prompt": "(Primitive, historically accurate, aged worn materials, clean textures, understated)"
    },
    {
        "name": "Ancient Maya",
        "prompt": "Ancient Mayan style {prompt}. Stone carvings, jungle scenes, hieroglyphs, temples, Pyramids, gold decor",
        "negative_prompt": "(Modern, digital, futuristic, cluttered, only text, shapes)"
    },
    {
        "name": "Nautical",
        "prompt": "Vintage nautical style {prompt}. Rope, ships at sea, maritime map, navigation tools, 19th century",
        "negative_prompt": "(Landlocked, urban city, futuristic, outer space, aerial view)"
    },
    {
        "name": "Bomber Jacket",
        "prompt": "Vintage bomber jacket style {prompt}. Distressed leather, shearling collar, embroidered patches, pop art",
        "negative_prompt": "(Low contrast, dull lighting, imperfections, asymmetrical, messy)"
    },
    {
        "name": "Samurai",
        "prompt": "Dramatic samurai style {prompt}. Armor, katana, dynamic action poses, ukiyo-e woodblock aesthetics",
        "negative_prompt": "(Gunfighters, lasers, sci-fi weapons, cluttered composition, only text)"
    },
    {
        "name": "American Traditional",
        "prompt": "Classic American traditional tattoo style {prompt}. Bold lines, nautical elements, retro flair, symbolic",
        "negative_prompt": "(Photorealistic, muted tones, intricate details, delicate lines, complex composition)"
    },
    {
        "name": "Propaganda Art",
        "prompt": "Vintage WWI propaganda poster {prompt}. Patriotic imagery, bold graphics, dramatic text, vivid colors",
        "negative_prompt": "(Delicate, intricate, hand-drawn, black and white, granular, impressionist)"
    },
    {
        "name": "Glitchcore",
        "prompt": "Glitchcore style {prompt}. Chaotic digital distortion, cyberpunk aesthetic, neon and chrome, fluorescent pop",
        "negative_prompt": "(Plain, static, orderly, structured, smooth, predictable, concrete)"
    },
    {
        "name": "Vaporgram",
        "prompt": "Vaporgram style {prompt}. Retro internet aesthetic, web 1.0 nostalgia, cyberpunk tones, neon lighting",
        "negative_prompt": "(Natural, matte, stone, metal, paper, analog, hand drawn)"
    },
    {
        "name": "Desaturated",
        "prompt": "Moody desaturated style {prompt}. Faded muted tones, grungy textures, gloomy lighting, post-apocalyptic",
        "negative_prompt": "(Vibrant, neon, saturated, high key, bright lighting, joyful)"
    },
    {
        "name": "Dieselpunk",
        "prompt": "Dieselpunk style {prompt}. Retrofuturism, art deco, mechanized armor, gritty adventurer aesthetic",
        "negative_prompt": "(Futuristic, sleek, clean, minimalist, synthetic materials, white)"
    },
    {
        "name": "My Little Pony",
        "prompt": "My Little Pony style {prompt}. Cute pastel ponies, vibrant color palette, happy fantastical land, friendship",
        "negative_prompt": "(Violent, scary, hyperrealistic, dirty, cluttered, creepy)"
    },
    {
        "name": "Ballet",
        "prompt": "Elegant ballet style {prompt}. Flowing tutus, graceful poses, dancers on stage, intricate movement, classical",
        "negative_prompt": "(Messy, unposed, candid, informal, street clothing, improvised)"
    },
    {
        "name": "Galactic",
        "prompt": "Majestic galactic style {prompt}. Vast colorful nebulae, stars and planets, sci-fi space themes, mystery",
        "negative_prompt": "(Earthly, familiar, mundane, manmade structures, human scale)"
    },
    {
        "name": "Streamer Bike",
        "prompt": "Colorful streamer bike {prompt}. Ribbons flowing, fun retro ride, vibrant summer vacation",
        "negative_prompt": "(Dark muted colors, austere minimal decor, sitting still)"
    },
    {
        "name": "Tropical Hotel",
        "prompt": "Luxe tropical hotel exterior {prompt}. Lush foliage, dramatic lighting, tiki torches, wooden textures",
        "negative_prompt": "(European medieval castle facade, gargoyles, stained glass)"
    },
    {
        "name": "Tiki Mug",
        "prompt": "Vintage tiki mug {prompt}. Ornate carvings, tropical motifs, bamboo textures, mysterious cocktail",
        "negative_prompt": "(Basic ceramic houseware, no ornamentation, filled with coffee)"
    },
    {
        "name": "Neon Racer",
        "prompt": "Futuristic neon racer bike {prompt}. Dynamic action pose, vibrant laser grid background",
        "negative_prompt": "(Realistic scale, pragmatic design, safety features, recreational bike)"
    },
    {
        "name": "Luau Fire Dancer",
        "prompt": "Mesmerizing luau fire dancer {prompt}. Fiery poi, flowing dress, beach sunset silhouette",
        "negative_prompt": "(Wearing winter clothes, snowy blizzard scene)"
    },
    {
        "name": "Day of the Dead",
        "prompt": "Vibrant Day of the Dead altar {prompt}. Marigolds, sugar skulls, candles, offerings, holiday decor",
        "negative_prompt": "(Muted tones, minimal decorations, empty background)"
    },
    {
        "name": "Sideshow Poster",
        "prompt": "Vintage sideshow poster {prompt}. Retro fonts, snake charmer, illustrated crowds, bright colors",
        "negative_prompt": "(Minimalist, text only, digital, flat graphics)"
    },
    {
        "name": "Vaporwave Graphics",
        "prompt": "Surreal vaporwave graphic design {prompt}. Glitch effects, retro pop art, neon palm trees",
        "negative_prompt": "(Traditional medium, paint strokes visible, brooding somber palette)"
    },
    {
        "name": "Voodoo Ceremony",
        "prompt": "A dark voodoo ceremony {prompt}. Chanting, offerings, occult artifacts, flickering firelight",
        "negative_prompt": "(Broad daylight, empty room, tidy clean room)"
    },
    {
        "name": "Blacklight Poster",
        "prompt": "Psychedelic blacklight poster {prompt}. Day-glo ink, hippie motifs, far out 60s art",
        "negative_prompt": "(Conservative palette, intricate details, visually dense composition)"
    },
    {
        "name": "Teslapunk",
        "prompt": "Teslapunk portrait {prompt}. Dramatic hairstyle, electrified coils, steampunk and neon vibes",
        "negative_prompt": "(Natural hair, simple clothing, basic poses)"
    },
    {
        "name": "Cassette Bedroom",
        "prompt": "Vaporwave bedroom scene {prompt}. Grid wallpaper, retro tech, nostalgic memorabilia",
        "negative_prompt": "(Empty room, muted lighting, banal decor)"
    },
    {
        "name": "Tropical Bathroom",
        "prompt": "Bohemian tropical bathroom {prompt}. Hanging plants, carved wood accents, woven textures",
        "negative_prompt": "(Cold tiles, chrome finishes, harsh lighting)"
    },
    {
        "name": "Voodoo Shop",
        "prompt": "Mysterious voodoo shop {prompt}. Occult artifacts, dried herbs, shadowy lighting, magic ingredients",
        "negative_prompt": "(Corporate retail store, tidy shelves, smiling staff)"
    },
    {
        "name": "Vintage Halloween",
        "prompt": "Creepy vintage Halloween {prompt}. Jack-o-lanterns, trick or treaters, creepy costumes, fall leaves",
        "negative_prompt": "(Minimalist flat graphics, digital art, random shapes/text)"
    },
    {
        "name": "Goth Boudoir",
        "prompt": "Moody gothic boudoir {prompt}. Black lace, candelabra, velvet chaise, romantic and haunting",
        "negative_prompt": "(Soft pastels, bright lighting, smiling happy expression)"
    },
    {
        "name": "Island Luau",
        "prompt": "Vibrant Hawaiian luau {prompt}. Tropical foliage, tiki torches, leis, flower crowns, beautiful sunset",
        "negative_prompt": "(Bundled figures, overcast sky, winter party accessories)"
    },
    {
        "name": "Tiki Outdoor Shower",
        "prompt": "Tropical outdoor shower {prompt}. Tropical foliage, bamboo, volcanic stone, resort vibes",
        "negative_prompt": "(Snowy scene, bare trees, closed down for winter)"
    },
    {
        "name": "Black Velvet Painting",
        "prompt": "Kitschy black velvet painting {prompt}. Glowing colors, Elvis portrait, religious iconography",
        "negative_prompt": "(Conservative palette and subject, perfectly centered, visually sparse)"
    },
    {
        "name": "Tattoo Print",
        "prompt": "Traditional tattoo art print {prompt}. Vibrant colors, retro flair, snake, nautical star, rose",
        "negative_prompt": "(Fine art, graduated tones, earthy color palette)"
    },
    {
        "name": "Addams Family",
        "prompt": "The Addams Family portrait {prompt}. Macabre, kooky, mysterious, and spooky family",
        "negative_prompt": "(Stiff formal portrait, conservative clothes, plain background)"
    },
    {
        "name": "Cassette Wall",
        "prompt": "Vaporwave cassette wall display {prompt}. Grid-style arrangement, retro tapes, 80s vibes",
        "negative_prompt": "(Frameless, uneven arrangement, varied tape conditions, cluttered)"
    },
    {
        "name": "Neon Tokyo",
        "prompt": "Vibrant neon Tokyo street {prompt}. Pink and blue hues, crowded shops, retro cabs, manga ads",
        "negative_prompt": "(Sepia tones, overcast lighting, dull concrete, empty streets)"
    },
    {
        "name": "Voodoo Queen",
        "prompt": "Mysterious voodoo queen portrait {prompt}. Dark purple lighting, occult artifacts, power",
        "negative_prompt": "(Business professional attire, office setting)"
    },
    {
        "name": "Surf Wood Sign",
        "prompt": "Vintage surf wood sign {prompt}. Ocean vibes, distressed paint, shark silhouette, route marker",
        "negative_prompt": "(Slick plastic material, bold solid colors, minimalist design)"
    },
    {
        "name": "Haunted Carnival",
        "prompt": "Abandoned haunted carnival {prompt}. Silent rides, eerie lights, fog, overgrown, spooky vibe",
        "negative_prompt": "(Broad daylight, opened and crowded, vibrant primary colors)"
    },
    {
        "name": "Tiki Idol",
        "prompt": "Foreboding tiki idol {prompt}. Volcanic rock, tribal carvings, jungle lighting, ominous",
        "negative_prompt": "(Cute whimsical style, delighted facial expression, bright colors)"
    },
    {
        "name": "Mall Goth",
        "prompt": "90s mall goth portrait {prompt}. Dark academia outfits, smoky eyeshadow, melancholy gaze",
        "negative_prompt": "(Natural skin and hair, pink pastel clothes, smiling happily)"
    },
    {
        "name": "Volcano Lair",
        "prompt": "Sinister volcano lair {prompt}. Bubbling lava, steel interior, control room, supervillain",
        "negative_prompt": "(Cluttered basement, leaky pipes, dust and cobwebs)"
    },
    {
        "name": "Graffiti Style",
        "prompt": "Wild graffiti style {prompt}. Vibrant spray paint art, dripping colors, bubble letters",
        "negative_prompt": "(Conservative palette, soft blended tones, ornate decorative details)"
    },
    {
        "name": "Impressionism",
        "prompt": "Impressionist painting {prompt} . Loose brushwork, vivid color, captivating light, modern scenes, everyday subjects, candid snapshots, plein air style"
    },
    {
        "name": "Fauvism",
        "negative_prompt": "Subtle muted tones, drab color palette, restrained brushwork, highly realistic portrayal",
        "prompt": "Fauvist painting {prompt} . Wild beast palette, intensely vivid colors, bold brushwork, expressive interpretation over realism"
    },
    {
        "name": "Dada",
        "prompt": "Dadaist artwork {prompt} . Nonsensical, absurdist, intentionally irrational, provocative, chaotic, anti-art establishment",
        "negative_prompt": "Literal representation, realism, order, harmony, deference to artistic conventions"
    },
    {
        "name": "Cubism",
        "prompt": "Cubist artwork {prompt} . Geometric planes, abstracted objects, shifting perspectives, simultaneous viewpoints",
        "negative_prompt": "Curved organic forms, identifiable objects, singular perspective, strict realism"
    },
    {
        "name": "Expressionism",
        "prompt": "Expressionist painting {prompt} . Subjective distortion in service of inner expression, using vivid color and energetic mark-making to convey angst and alienation of modern experience",
        "negative_prompt": "Realism, placid refinement, conventional harmony, everyday bourgeois existence"
    },
    {
        "name": "Surrealism",
        "prompt": "Surrealist painting {prompt} . Dream imagery, jarring juxtapositions, subconscious thoughts, fantastical subjects, automatism",
        "negative_prompt": "Realism, mundane objects, conscious control, plausibility, traditional methods"
    },
    {
        "name": "Symbolism",
        "prompt": "Symbolist painting {prompt} . Suggestive, evocative, meanings beyond physical forms, mystical, spiritual themes",
        "negative_prompt": "Literalism, physicality, scientific rationalism, skepticism, secular subjects"
    },
    {
        "name": "Op Art",
        "prompt": "Op Art painting {prompt} . Optical vibrating effects, disorienting movement illusion, flashing pulses, dazzling viewers",
        "negative_prompt": "Still contemplation, solid masses, mute subtlety, sensory rest"
    },
    {
        "name": "Minimalism",
        "prompt": "Minimalist artwork {prompt} . Plain, simple geometric forms, limited color, deliberate restraint, clean lines",
        "negative_prompt": "Elaborate decoration, complex subjects, vibrant polychromatic palette, exuberant mark-making"
    },
    {
        "name": "Conceptual Art",
        "prompt": "Conceptual artwork {prompt} . Idea takes precedence over aesthetics, provokes intellectual interpretation",
        "negative_prompt": "Strictly visual appeal, no symbolic meaning, purely retinal response"
    },
    {
        "name": "Folk Art",
        "prompt": "Folk artwork {prompt} . Raw directness reflecting everyday life, unschooled traditionalism passed through community, urgency of unpretentious expression",
        "negative_prompt": "Ivory tower theory, institutional dogma, elite gatekeeping, exclusion of those deemed uncultured"
    },
    {
        "name": "Naive Art",
        "prompt": "Naive art style {prompt} . Untrained simplicity, childlike perspective, flattened space, whimsical subjects",
        "negative_prompt": "Visual mastery, airbrush techniques, atmospheric perspective, serious topics"
    },
    {
        "name": "Outsider Art",
        "prompt": "Outsider artwork {prompt} . Unfiltered personal vision, marginalized artists outside the mainstream, mediums chosen for expressiveness not formal qualities",
        "negative_prompt": "Pandering to popular taste, gentrified high art sensibilities, conditioned by academy conventions"
    },
    {
        "name": "Photorealism",
        "prompt": "Photorealistic painting {prompt} . Virtuosic rendering elevates technical skill in service of duplicating photography's visual facts through painterly verisimilitude",
        "negative_prompt": "Loose expressionist brushwork, emotional extrapolation from reality, filtered truth"
    },
    {
        "name": "Suprematism",
        "prompt": "Suprematist composition {prompt} . Abstract geometric forms, centered balance, planes of color, purity through reduction",
        "negative_prompt": "Representational elements, off-center asymmetry, pictorial space, elaborate detail"
    },
    {
        "name": "De Stijl",
        "prompt": "De Stijl style {prompt} . Strict geometry, primary colors, white ground, form reduced to verticals, horizontals, rectangles",
        "negative_prompt": "Loose organic shapes, muted secondary hues, textured surfaces, diagonal elements"
    },
    {
        "name": "Bauhaus",
        "prompt": "Bauhaus style {prompt} . Geometric simplicity, function over ornament, rejection of bourgeoisie decadence, truth to materials",
        "negative_prompt": "Decorative embellishments, opulent aesthetics, art for art's sake, disguising mediums"
    },
    {
        "name": "Constructivism",
        "prompt": "Constructivist composition {prompt} . Geometric abstraction, architectural arrangement, dynamism, utopian modernity, social purpose",
        "negative_prompt": "Mimesis, decorative ornament, static equilibrium, reactionary nostalgia, art for art's sake"
    },
    {
        "name": "Futurism",
        "prompt": "Futurist painting {prompt} . Fast moving energy, technological triumph, avant-garde disruption, destroy past traditions",
        "negative_prompt": "Slow pace, handmade craft, quaint historical revivalism, preservation of academic standards"
    },
    {
        "name": "Rayonism",
        "prompt": "Rayonist painting {prompt} . Abstract movement, luminous radiating lines, dynamic sensation of speed",
        "negative_prompt": "Solid mass, muted palette, motionless stasis, photographic realism"
    },
    {
        "name": "Vorticism",
        "prompt": "Vorticist composition {prompt} . Angular lines, hard-edged shapes, machine aesthetic, vortex of abstract energy",
        "negative_prompt": "Organic curves, figural allusion, handcrafted technique, passive tranquility"
    },
    {
        "name": "Orphism",
        "prompt": "Orphist painting {prompt} . Luminous color, abstract mythic symbolism, non-objective, pushes art into pure idea",
        "negative_prompt": "Earthbound representation, scientific naturalism, commonplace reality, pedestrian subjects"
    },
    {
        "name": "Der Blaue Reiter",
        "prompt": "Der Blaue Reiter style {prompt} . Primitivism, spiritual yearning, Wassily Kandinsky abstract compositions"
    },
    {
        "name": "Die Brücke",
        "negative_prompt": "Refinement, decorum, placidity, adherence to artistic traditions",
        "prompt": "Die Brücke style {prompt} . Raw expressionism, graffiti-like colors, emotional intensity, rejection of conventions"
    },
    {
        "name": "Ashcan School",
        "prompt": "Ashcan School style {prompt} . Unvarnished urban realism, working class subjects, steerage immigration, street scenes"
    },
    {
        "name": "Hudson River School",
        "negative_prompt": "Urban industry, cultivated farmlands, overcast mundane atmosphere",
        "prompt": "Hudson River School landscape {prompt} . Majesty of untamed nature, wilderness vistas, luminous romantic skies"
    },
    {
        "name": "Luminism",
        "prompt": "Luminist landscape {prompt} . Radiant light effects, tranquil mood, precise details emerge from mist",
        "negative_prompt": "Gloom, discordance, loosely brushed suggestion, storms and darkness"
    },
    {
        "name": "Tonalism",
        "prompt": "Tonalist landscape {prompt} . Hazy soft-edged forms, quiet serenity, subdued tones, somber feelings",
        "negative_prompt": "Hard outlines, exuberance, brilliant colors, surface distraction"
    },
    {
        "name": "Barbizon School",
        "prompt": "Barbizon School landscape {prompt} . Natural grandeur, plein air style, living quality of light, oneness with nature",
        "negative_prompt": "Urban settings, artificial indoor lighting, separate from environment, humanized landscape"
    },
    {
        "name": "Academic Art",
        "prompt": "Academic artwork {prompt} . High Renaissance techniques, idealized beauty, historical/mythological topics, formal studio training",
        "negative_prompt": "Mediocre execution, unconventional aesthetics, ordinary subjects, outsider artists"
    },
    {
        "name": "Rococo",
        "prompt": "Rococo interior {prompt} . Ornate gold detailing, light and frothy, delicate asymmetry and pastel colors, aristocratic luxury"
    },
    {
        "name": "Neoclassicism",
        "negative_prompt": "Decadence, frailty, everyday people, romantic organicism",
        "prompt": "Neoclassical painting {prompt} . Severe geometry, stoic virtues, heroic figures, Greco-Roman classicism revival"
    },
    {
        "name": "Romanticism",
        "prompt": "Romantic painting {prompt} . Sublime awe of nature, storms, ruins, extremes of emotion, imagination unrestrained by reason",
        "negative_prompt": "Supervised society, complacent daily life, calm stability, rational control"
    },
    {
        "name": "Realism",
        "prompt": "Realist painting {prompt} . Unvarnished truth without idealization, peasantry and working class subjects, contemporary life",
        "negative_prompt": "Gloss of perfection, aristocrats and gods, exotic fantasy, mythological tableaus"
    },
    {
        "name": "Social Realism",
        "negative_prompt": "Propaganda, political repression, censorship, kitsch conventMice, one-dimensional stereotyping",
        "prompt": "Social Realism painting {prompt} . Unflinching witness to hardship and adversity faced by ordinary people, borne with deep empathy yet refusal to romanticize"
    },
    {
        "name": "Plein Air",
        "prompt": "Plein air landscape {prompt} . Open-air painting, direct communion with nature, vivid effects of outdoor light",
        "negative_prompt": "Contrived studio sets, artificial lighting, modifications from sketches"
    },
    {
        "name": "Pre-Raphaelite",
        "prompt": "Pre-Raphaelite painting {prompt} . Medieval and mythological scenes, lush colors, subjects imbued with emotional intensity",
        "negative_prompt": "Contemporary settings, reserve, factual accuracy over feelings"
    },
    {
        "name": "Art Informel",
        "prompt": "Art Informel painting {prompt} . Spontaneous gesture, lyrical abstraction, unrestrained directness of brushwork and color",
        "negative_prompt": "Meticulous craftsmanship, figural realism, premeditated approach"
    },
    {
        "name": "Pure Typography",
        "prompt": "Lettrist artwork {prompt} . Avant-garde letters freed into pure form, visual rhythm and texture of typography as sole communication",
        "negative_prompt": "Readability, coherence, concrete meaning beyond visual experience"
    },
    {
        "name": "Hard-edge Painting",
        "prompt": "Hard-edge abstract painting {prompt} . Crisp optical color shapes, clean smooth technique, sharp uniform lines",
        "negative_prompt": "Painterly texture, irregular forms, smudged blurred edges"
    },
    {
        "name": "Geometric Abstraction",
        "prompt": "Geometric abstract painting {prompt} . Lines, circles, triangles, grids in dynamic tension, pure color, flatness",
        "negative_prompt": "Blobs, splatters, curlicues, decorative illusion of depth"
    },
    {
        "name": "Lyrical Abstraction",
        "prompt": "Lyrical abstract painting {prompt} . Paint itself as subject, pigment's resonance, suggestion over definition, pleasure of free brushed color",
        "negative_prompt": "Descriptive contours, tonal modeling, ostensible subjects beyond the paint"
    },
    {
        "name": "Post-Painterly Abstraction",
        "prompt": "Post-Painterly abstract painting {prompt} . Open fields of color, stain painting, raw canvas, expansive flat forms",
        "negative_prompt": "Dense composition, decorative effects, illusionism, poured rivulets"
    },
    {
        "name": "Kinetic Art",
        "prompt": "Kinetic sculpture {prompt} . Moving parts, plays of light, floating shapes, real motion that transforms itself"
    },
    {
        "name": "Land Art",
        "negative_prompt": "Discrete portable work, artificial substances, handheld size, designed for institutional exhibition",
        "prompt": "Land Art sculpture {prompt} . Sculptural manipulation of the landscape itself, raw natural materials, grand scale, rejecting galleries and museums"
    },
    {
        "name": "Performance Art",
        "prompt": "Performance Art {prompt} . Actions staged by artist, bodily experience, ephemeral real-time situations, challenging ideas of artistic object and ownership",
        "negative_prompt": "Formed by assistants, cerebral theory, permanent object, saleable commodity"
    },
    {
        "name": "Installation Art",
        "prompt": "Installation artwork {prompt} . Viewer immersion in constructed situation, spatial narrative, total artwork environment",
        "negative_prompt": "Individual element, no enveloping concept, traditional display on wall or pedestal"
    },
    {
        "name": "Video Art",
        "prompt": "Video Art {prompt} . Television as artistic medium, technology art laying bare vocabularies of commercial television and cinema through experimentation",
        "negative_prompt": "Passive entertainment, narrative immersion, transparency of medium, uniqueness of traditional object art"
    },
    {
        "name": "Digital Art",
        "prompt": "Digital artwork {prompt} . Computer as artistic medium, networked transmission, remixing hypermedia, playing with freedoms and limitations of software",
        "negative_prompt": "Intuitive handmade craft, auratic object, circumscribed by physical studio space, permanence of traditional materials"
    },
    {
        "name": "New Media Art",
        "prompt": "New Media artwork {prompt} . Radical redefinitions through cutting-edge tools, participatory and interactive modes, networked digital context, criticizing old media paradigms",
        "negative_prompt": "Ossified assumptions, art institutionalized, worship of relics, passive viewing of rarefied objects"
    },
    {
        "name": "Street Art",
        "prompt": "Street art {prompt} . Public space as canvas, satirical subversion, reclaiming visual territory, guerilla art, ephemeral treasures",
        "negative_prompt": "Institutional settings, pious decorum, officially sanctioned statements, law and order"
    },
    {
        "name": "Stuckism",
        "prompt": "Stuckist painting {prompt} . Expressive directness and poetic human vision as antidote to ironic postmodernism of establishment artworld",
        "negative_prompt": "Faceless ideology, impenetrable critical theory, dematerialized art object, institutional gamesmanship"
    },
    {
        "name": "Lowbrow Art",
        "prompt": "Lowbrow artwork {prompt} . Underground irreverence, psychedelic counterculture, not highbrow but subversive punk energy",
        "negative_prompt": "Highbrow elitism, exclusivity, insider discourse, needless obscurantism"
    },
    {
        "name": "Photomontage",
        "prompt": "Photomontage artwork {prompt} . Composite photographed elements make improbable combinations, unexpected contexts",
        "negative_prompt": "Seamless scenes, plausibility, faithful representation, discretion"
    },
    {
        "name": "Diorama",
        "prompt": "Diorama artwork {prompt} . Imaginative lifelike 3D scenes in miniature, peaceful temples or violent battles enacted",
        "negative_prompt": "Monochrome flatness, larger than life scale, void spaces, nothing happening"
    },
    {
        "name": "Assemblage",
        "prompt": "Assemblage sculpture {prompt} . Found objects unified into an invented whole, unexpected juxtapositions, witty and uncanny"
    },
    {
        "name": "Combine Painting",
        "negative_prompt": "Flat canvas, unified medium, paint contained on surface, discreet frame",
        "prompt": "Combine painting {prompt} . Multimedia collage combines painting and objects protruding into real space"
    },
    {
        "name": "Happenings",
        "prompt": "Happening artwork {prompt} . Early performance art, spontaneous actions before gathered audience, anarchic ephemerality",
        "negative_prompt": "Rehearsed and repeatable, the contextless artifact, institutional preservation, order"
    },
    {
        "name": "Mail Art",
        "prompt": "Mail Art {prompt} . Decentralized art movement using postal service as medium, collage, rubber stamps, artist networks",
        "negative_prompt": "Precious unique object, high ticket prices, elite galleries, individual glory"
    },
    {
        "name": "Neo-Dada",
        "prompt": "Neo-Dada artwork {prompt} . Modern resurrection of absurdism and anarchy, derailing rational thought through chance procedures, irrational juxtapositions, deliberately ridiculous",
        "negative_prompt": "Sober good taste, rational composition, clear meaning, deference to artistic ideals"
    },
    {
        "name": "Neo-Expressionism",
        "prompt": "Neo-Expressionist painting {prompt} . Spontaneous process over premeditation, raw primal gestural brushwork, intense colors, mythic archetypes, reborn painting connects to authentic human experience",
        "negative_prompt": "Ideology, mechanical reproduction, oblique abstraction, postmodernist irony"
    },
    {
        "name": "Bad Painting",
        "prompt": "Bad Painting {prompt} . Defiant amateurishness as assault on pretense, calculated rejection of good taste, genius in apparently sloppy improvisation",
        "negative_prompt": "Slick empty proficiency, flawless mediocrity, deaf to lived experience"
    },
    {
        "name": "Graffiti",
        "prompt": "Graffiti artwork {prompt} . Bold tags claiming public space, street writing as folk art of urban environment, anti-establishment energy"
    },
    {
        "name": "Traditional Figurative Art",
        "prompt": "Traditional figurative artwork {prompt} . Representational painting and sculpture, exaltation of human form, rediscovery of visual storytelling",
        "negative_prompt": "Non-objective art, conceptualStatement, aversion to skill and aesthetics"
    },
    {
        "name": "Classical Realism",
        "prompt": "Classical Realist portrait {prompt} . Traditional atelier training, emphasis on patient craft and technical finesse, contemporary manifestation of academic painting",
        "negative_prompt": "Edgy avant-gardism, attacking conventions, purposely shocking, rebelling against foundations"
    },
    {
        "name": "Contemporary Realism",
        "prompt": "Contemporary Realist painting {prompt} . Penetratingly observed authenticity without idealization, conveying subtle truths through martially honed skills of seeing and execution"
    },
    {
        "name": "Hyperrealism",
        "prompt": "Hyperrealistic painting {prompt} . Visual clarity magnified beyond human eye's abilities, hallucinatory detail exceeding real perceptive capacities",
        "negative_prompt": "Suggestion and omission, stripped-down abstraction, reduced to core meaning"
    },
    {
        "name": "Magic Realism",
        "prompt": "Magic Realist painting {prompt} . Matter-of-fact presentation of the fantastical, seamless merging of material reality and flights of fancy, precision of illusion",
        "negative_prompt": "Suspension of disbelief, maintaining distinction between worlds, respecting limitations, plausible realism"
    },
    {
        "name": "New Objectivity",
        "prompt": "New Objectivity painting {prompt} . Detached precisionist scrutiny, social criticism through realism's exacting close description of reality's imperfections",
        "negative_prompt": "Romantic souls, gauzy impressionism, glossing over flaws, idealized distortions"
    },
    {
        "name": "Precisionism",
        "negative_prompt": "Tangled underbrush, weathered ruins, irregular forms, gloomy disarray",
        "prompt": "Precisionist painting {prompt} . Industrial landscapes, gleaming architecture, geometric purity, crisp lines, urban order"
    },
    {
        "name": "Figurative Expressionism",
        "prompt": "Figurative Expressionist painting {prompt} . First-generation abstract expressionists later returned to the human form, with same intense impulsive brushwork and boldness of color",
        "negative_prompt": "Refusal of representation, ascetic purity of total abstraction, fear of narrative content"
    },
    {
        "name": "Neue Wilde",
        "prompt": "Neue Wilde painting {prompt} . Raw expressionistic brushwork, garish palette, rebellious exuberance, punk sensibility",
        "negative_prompt": "Apollonian calm, good taste, deference to artistic ideals, gentility"
    },
    {
        "name": "New Perpendicular art",
        "prompt": "New Perpendicular painting {prompt} . Jagged lines, bright colors, erratic brush, postmodernist combines early modern art styles",
        "negative_prompt": "Controlled geometry, unsaturated hues, even smoothness, revivalism"
    },
    {
        "name": "New Simplicity",
        "prompt": "New Simplicity painting {prompt} . Return to naive style and representational tradition, allegorical and mythic themes, rejects avant-garde fragmentation",
        "negative_prompt": "Childish technique, contemporary subjects, cutting-edge experimentation"
    },
    {
        "name": "Remodernism",
        "negative_prompt": "Warmed over postmodernist irony, demoralized alienation, ideology excusing disconnection from meaningful making",
        "prompt": "Remodernist painting {prompt} . Spiritual meaning, disruptive innovation, authenticity and radical imagination to reconnect art with human experience"
    },
    {
        "name": "Norwegian romantic nationalism",
        "prompt": "Norwegian romantic landscape {prompt} . Rugged natural beauty, folk legends, historic traditions, sublime Nordic mood",
        "negative_prompt": "Urban modernity, cosmopolitan trends, flat topography, quaint pastoral images"
    },
    {
        "name": "Socialist Realism",
        "prompt": "Socialist Realist painting {prompt} . Heroic archetypes glorifying the utopian worker's state, figures with idealized physiques posed nobly in propaganda images",
        "negative_prompt": "Bourgeois decadence, exposé of harsh conditions, individual diversity"
    },
    {
        "name": "Propaganda Poster Art",
        "prompt": "Propaganda poster {prompt} . Bold graphics, simple message, stark contrasts, motivational aesthetics, unambiguous imagery glorifying party goals",
        "negative_prompt": "Understatement, detail, nuance, quiet sympathy for outsiders, anything demotivating or distracting from clear message"
    },
    {
        "name": "Heroic Realism",
        "prompt": "Heroic Realist sculpture {prompt} . Figures with idealized musculature posing dynamically, celebrating ideals of mass utopian culture",
        "negative_prompt": "Spindly awkward bodies, static postures, unique personalized details"
    },
    {
        "name": "Naïve Art",
        "prompt": "Naive art painting {prompt} . Untrained style, candid perspective, abandoning tricks of realism, expressively handmade, unpretentious charm",
        "negative_prompt": "Academic refinement, high-polish finish, mechanical perfection, digital tools removing artist's hand, sterile presentation"
    },
    {
        "name": "Art Brut",
        "prompt": "Art Brut artwork {prompt} . Unfiltered creative force, unconcerned with artistic norms or mainstream sensibility, mediums chosen for expressiveness not formal qualities",
        "negative_prompt": "Reserved professionalism, formulaic fetching styles, predictable taste-driven mannerisms, aiming to impress cognoscenti"
    },
    {
        "name": "Neo-primitivism",
        "prompt": "Neo-primitivist painting {prompt} . Masks, artifacts, ceremonial objects, references to ancestral magic and myths, abandoning Western rationalism",
        "negative_prompt": "Familiar interior scenes, technology, pop culture materialism, casual snapshots, convenience"
    },
    {
        "name": "Visionary Art",
        "prompt": "Visionary artwork {prompt} . Mystical emanations and spiritual revelations, unchained imagination, profoundly personal symbolism",
        "negative_prompt": "Pedestrian subjects, everyday materialism, replicating physical appearances, casual improvisation"
    },
    {
        "name": "Intuitive Art",
        "prompt": "Intuitive artwork {prompt} . Surrendered to creative flow state, unconsciously manifested, direct transmission of inspiration through sensibility developed outside thought",
        "negative_prompt": "Conceptual premeditation, theory-minded planning, diagrams and preparatory studies, cerebral editing"
    },
    {
        "name": "Pseudorealism",
        "prompt": "Pseudorealistic painting {prompt} . Illusion of impossible clarity and tangibility, 3D rendering of imagined perfection, realer than real, supernatural focus",
        "negative_prompt": "Hazy soft-focus, stripped-down reduction, blurred vagueness suggesting essence"
    },
    {
        "name": "Radical Realism",
        "prompt": "Radical Realist still life {prompt} . Ordinary subjects elevated through supernatural lighting effects, tableaus from unfamiliar angles, everyday epiphanies",
        "negative_prompt": "Extraordinary occurrences, elaborate symbolic arrangements, viewing things conventionally"
    },
    {
        "name": "Critical Realism",
        "prompt": "Critical Realism painting {prompt} . Incisive social analysis conveyed through dispassionate realist scrutiny, shining flashlight objectivity on unpleasant realities",
        "negative_prompt": "Romantic distortions, gauzy illusions, belying inconvenient truths"
    },
    {
        "name": "Neue Sachlichkeit",
        "prompt": "Neue Sachlichkeit portrait {prompt} . Psychological intensity conveyed through detached hyperrealist scrutiny, social masks penetrated in moments of unguarded emotion"
    },
    {
        "name": "Regionalism",
        "prompt": "Regionalist painting {prompt} . Romanticized heartland subjects, nostalgia for small town life, realist exaltation of provincial America",
        "negative_prompt": "Cosmopolitanism, cultural critique, European avant-garde, modernist innovation"
    },
    {
        "name": "Abstract Expressionism",
        "prompt": "Abstract Expressionist painting {prompt} . Spontaneous paint handling, energetic mark-making, color symbolism, monumental scale",
        "negative_prompt": "Precise illusionistic realism, miniaturism, delicate controlled handling, depicting identifiable objects"
    },
    {
        "name": "Later European abstraction",
        "prompt": "Later European abstract painting {prompt} . Lyrical informalism, matiére, tachisme, arte informel, cobra group freedoms",
        "negative_prompt": "Stringent conceptualism, systems art, technological medium, American minimalism"
    },
    {
        "name": "Tachisme",
        "prompt": "Tachiste painting {prompt} . Spontaneous expressive brushwork, dynamic swirling paint traces, process over preconception",
        "negative_prompt": "Meticulous planning, diagrammatic underdrawing, reserved color, static composition"
    },
    {
        "name": "Abstraction-Création",
        "prompt": "Abstraction-Création artwork {prompt} . Geometric abstraction, architectonic line patterns, mathematically harmonious composition",
        "negative_prompt": "Organic asymmetry, goopy paint, introspective emotionalism, 'landscapes of the soul'"
    },
    {
        "name": "Gutai",
        "negative_prompt": "Asymmetries, introspective emotionalism, indulgent painterliness, 'landscapes of the soul'",
        "prompt": "Gutai artwork {prompt} . Radical experimentalism using real spaces and time, spontaneous event and performance, forerunner to fluxus and happenings"
    },
    {
        "name": "British Pop Art",
        "negative_prompt": "High art pretensions, existential angst, abstract attacks on institution",
        "prompt": "British Pop Art {prompt} . Mass media, pop culture, irony and appropriation questioning consumerism and commodification"
    },
    {
        "name": "Situationist International",
        "prompt": "Situationist artwork {prompt} . Bureaucratized spectacle upended through disruptive interventions, détournement of consumer images",
        "negative_prompt": "Complacency, acquiescence to capitalism, individualism, art for art's sake"
    },
    {
        "name": "Lettrism",
        "prompt": "Lettrist artwork {prompt} . Avant-garde letters liberated into pure form, only typeface communicating, counter prevailing language systems",
        "negative_prompt": "Readability, coherence, grammar, concrete meaning beyond visual experience"
    },
    {
        "name": "St Ives School",
        "prompt": "St Ives School painting {prompt} . Luminous landscape and seascape, postwar British gestural modernism",
        "negative_prompt": "Social realism, urban alienation, pop art commercialism"
    },
    {
        "name": "Transavantgarde",
        "prompt": "Transavantgarde painting {prompt} . Exuberant painterly expression, mythic archetypes, returns to painting and figuration with irony in postmodern era",
        "negative_prompt": "Rejection of illusion, dematerialization, oblique abstraction, ideology, institutional critique"
    },
    {
        "name": "Transgressive Art",
        "prompt": "Transgressive artwork {prompt} . Shocking confrontational motifs violate social taboos, extreme fringe imagery scorns mainstream taste",
        "negative_prompt": "Inoffensive politeness, pandering to respectable opinion, family-friendly pablum"
    },
    {
        "name": "Neo Pop",
        "negative_prompt": "High art elitism, exclusivity, aloof obscurantism, saying nothing to anyone",
        "prompt": "Neo-Pop artwork {prompt} . Vibrant postmodern remixing of commercial pop culture, hyperreality of simulation, larger than life appropriated imagery"
    },
    {
        "name": "Kitsch Movement",
        "prompt": "Kitsch artwork {prompt} . Exaltation of tacky mass-produced ornament, garish colors and exaggerated gesture, vulgarly displaying bad taste",
        "negative_prompt": "Sophistication appeasing high art sensibilities, understatement pandering to privileged gatekeepers"
    },
    {
        "name": "Virtual Art",
        "prompt": "Virtual artwork {prompt} . Full immersion in responsive simulated environments, navigating imaginary spaces through avatars and AI systems",
        "negative_prompt": "Physical art object, passive viewership, discrete artwork, real world constraints"
    },
    {
        "name": "Internet Art",
        "prompt": "Internet artwork {prompt} . Browser as venue, leveraging networked interconnectivity as artistic material, natively digital art in the expanded space of hypertext",
        "negative_prompt": "Physical gallery, art market economics, traditional assumptions of discrete object, artist as individual genius"
    },
    {
        "name": "Computer Art",
        "prompt": "Computer art {prompt} . Code as creative material, programmed algorithms generating emergent artworks, computational processes manifesting unimagined permutations",
        "negative_prompt": "Intuitive handmade art, injecting irregularity, chance, and happy imperfections"
    },
    {
        "name": "Information Art",
        "prompt": "Information artwork {prompt} . Innovative aesthetics of data visualization, statistics and databases as medium, patterns from abstract sources given tangibility",
        "negative_prompt": "Unique handmade object, raw artistic gesture, auratic one-of-a-kind original"
    },
    {
        "name": "Systems Art",
        "prompt": "Systems artwork {prompt} . Programmatic art executed according to plan, introducing causality through algorithms and logical processes into organic art"
    },
    {
        "name": "Bio Art",
        "negative_prompt": "Inanimate materials, artificial toxic substances, destructive lack of imagination blindly following industrial norms",
        "prompt": "BioArt {prompt} . Manipulates living matter and organisms as artistic medium, new natures carefully cultivated by artist-scientist"
    },
    {
        "name": "Genetic Art",
        "prompt": "Genetic artwork {prompt} . Recombinant DNA aesthetics, gene sequences modified for artistic purposes, radical merger of technology and the organic",
        "negative_prompt": "Fearful Ludditism, thoughtless assumptions, dangerous superstitions, blind allegiance to tradition"
    },
    {
        "name": "Sustainable Art",
        "prompt": "Sustainable artwork {prompt} . Eco-aware methods, renewables and ephemeral materials provoking consciousness, instructive experiences",
        "negative_prompt": "Profligate waste, reckless pollution, future-blind harm, toxicity"
    },
    {
        "name": "Interactive Art",
        "prompt": "Interactive artwork {prompt} . Audience inputs determine permutations, each encounter unique, empowering participant creativity",
        "negative_prompt": "Passive viewership, immutable object, sterile remove, rarefied untouchable artifact"
    },
    {
        "name": "Video Games",
        "prompt": "Video game artwork {prompt} . Virtual interactive environments as artistic material, playable narrative situations expanding culture's toolkit",
        "negative_prompt": "Non-immersive media, impermeable works, inhibiting audience involvement, elite proprietary exhibition"
    },
    {
        "name": "Machinima",
        "prompt": "Machinima artwork {prompt} . Cinematic narratives constructed real-time in 3D game engines, liberating filmmaking from industrial barriers"
    },
    {
        "name": "Artware",
        "negative_prompt": "Compliant conformity, dutifully maintaining order, polite respect for business as usual",
        "prompt": "Artware {prompt} . Artistic viruses and malware as critique of corporatized digital space, injecting disruptive changes into proprietary systems"
    },
    {
        "name": "Demoscene",
        "prompt": "Demoscene artwork {prompt} . Dazzling real-time animations prove programming virtuosity, maximalist computer graphics demonstrate mastery of hardware",
        "negative_prompt": "Passive viewership, pedestrian standards, bland corporate templates"
    },
    {
        "name": "Math Art",
        "prompt": "Mathematical artwork {prompt} . Elegant complexities generated through iterations of computational processes and algorithms"
    },
    {
        "name": "Data Art",
        "prompt": "Data artwork {prompt} . Aesthetic lens on statistics and datasets, infovis, deriving cultural meaning from impersonal information",
        "negative_prompt": "Personal expression, anthropocentrism, internal mythology and symbolism"
    },
    {
        "name": "Plotter Art",
        "prompt": "Plotter artwork {prompt} . Ink illustrations precisely executed by drawing machine, drafting aesthetic, programmable art with tangible materiality",
        "negative_prompt": "Ephemerality of pixels, suspended disbelief of virtual realm removed from physical constraints"
    },
    {
        "name": "VR Art",
        "prompt": "VR artwork {prompt} . Artistic exploration of virtual reality, navigable 3D environments, full immersion in simulated spaces"
    },
    {
        "name": "AR Art",
        "negative_prompt": "Consensus reality, unaugmented ordinariness, confinement to mundane existence",
        "prompt": "AR artwork {prompt} . Overlaying digital onto physical through augmented reality, altered perceptions of familiar surroundings"
    },
    {
        "name": "Circuit Bending",
        "prompt": "Circuit bent artwork {prompt} . Hacked electronic instruments and devices exhibit unpredictable ornamental behaviors",
        "negative_prompt": "Single-minded utility, sterile function, closeted inside opaque cases, behaving just as programmed"
    },
    {
        "name": "DIY Art",
        "prompt": "DIY artwork {prompt} . Amateur embrace of direct hands-on creativity, improvised misuse of everyday materials, carefree unskilled experimentation"
    },
    {
        "name": "Maker Culture",
        "negative_prompt": "Mass manufactured consumerism, top-down paternalistic technocracy, culture of passive non-participation",
        "prompt": "Maker artwork {prompt} . Hands-on DIY engineering, customized odd inventions, quirky hobbyist tech crafts, decentralized non-professional innovation"
    },
    {
        "name": "Self-taught Art",
        "prompt": "Self-taught artwork {prompt} . Intuitive creative force, idiosyncratic practices invented without formal training, vision uncompromised by conformist technique",
        "negative_prompt": "Academic disciplines, foundation courses, assigned readings, mandatory prerequisites, grades"
    },
    {
        "name": "Neo-Pop",
        "prompt": "Neo-Pop artwork {prompt} . Exuberant remixing of commercial pop culture, hyper-reality of simulation, saturated colors, larger than life appropriated imagery",
        "negative_prompt": "High art elitism, exclusivity, aloof obscurantism, saying nothing to anyone"
    },
    {
        "name": "Young British Artists",
        "prompt": "Young British Artist artwork {prompt} . Sensationalism and controversy in your face, feigned indifference masking desperation to épater la bourgeoisie",
        "negative_prompt": "Sober analysis, constructive dialogue, integrity withoutNZ posturing"
    },
    {
        "name": "Appropriation",
        "prompt": "Appropriation artwork {prompt} . Familiar media content recontextualized, questioning originality and ownership in image saturated culture by strategic theft",
        "negative_prompt": "Celebrating branded individual genius, romantic mythology of personal vision, defending intellectual property"
    },
    {
        "name": "Algorithmic Art",
        "negative_prompt": "Top-down control, paternalistic technocracy, individualism stifled by hive mind",
        "prompt": "Algorithmic artwork {prompt} . Code as artist, set of programmed instructions extrapolated through permutations, creative emergence from logical process"
    },
]
