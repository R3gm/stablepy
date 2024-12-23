AUX_TASKS = [
    "HED",
    "Midas",
    "MLSD",
    "Openpose",
    "PidiNet",
    "NormalBae",
    "Lineart",
    "LineartAnime",
    "Canny",
    "ContentShuffle",
]

TRANSFORMERS_LIB_TASKS = ["DPT", "UPerNet", "ZoeDepth", "SegFormer", "DepthAnything"]

AUX_BETA_TASKS = ["TEED", "Anyline", "Lineart standard"]

EXTRA_AUX_TASKS = ["Recolor", "Blur"]

ALL_PREPROCESSOR_TASKS = AUX_TASKS + TRANSFORMERS_LIB_TASKS + AUX_BETA_TASKS + EXTRA_AUX_TASKS

T2I_PREPROCESSOR_NAME = {
    "sdxl_canny_t2i": "Canny",
    "sdxl_openpose_t2i": "Openpose core",
    "sdxl_sketch_t2i": "PidiNet",
    "sdxl_depth-midas_t2i": "Midas",
    "sdxl_lineart_t2i": "Lineart",
}

TASK_AND_PREPROCESSORS = {
    "openpose": [
        "Openpose",
        "Openpose core",
        "None",
    ],
    "scribble": [
        "HED",
        "PidiNet",
        "TEED",
        "None",
    ],
    "softedge": [
        "PidiNet",
        "HED",
        "HED safe",
        "PidiNet safe",
        "TEED",
        "None",
    ],
    "segmentation": [
        "UPerNet",
        "SegFormer",
        "None",
    ],
    "depth": [
        "DPT",
        "Midas",
        "ZoeDepth",
        "DepthAnything",
        "None",
    ],
    "normalbae": [
        "NormalBae",
        "None",
    ],
    "lineart": [
        "Lineart",
        "Lineart coarse",
        "Lineart (anime)",
        "Lineart standard",
        "Anyline",
        "None",
        "None (anime)",
    ],
    "lineart_anime": [
        "Lineart",
        "Lineart coarse",
        "Lineart (anime)",
        "Lineart standard",
        "Anyline",
        "None",
        "None (anime)",
    ],
    "shuffle": [
        "ContentShuffle",
        "None",
    ],
    "canny": [
        "Canny",
        "None",
    ],
    "mlsd": [
        "MLSD",
        "None",
    ],
    "ip2p": [
        "None"
    ],
    "recolor": [
        "Recolor luminance",
        "Recolor intensity",
        "None",
    ],
    "pattern": [
        "None",
    ],
    "tile": [
        "Blur",
        "None",
    ],
    "repaint": [
        "None",
    ],
}
