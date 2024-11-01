from diffusers import EulerDiscreteScheduler
import json
from huggingface_hub import hf_hub_download
from ..logging.logging_setup import logger
from .constants import (
    INCOMPATIBILITY_SAMPLER_SCHEDULE,
    SCHEDULE_TYPES,
    SCHEDULE_PREDICTION_TYPE,
    AYS_SCHEDULES,
)


def configure_scheduler(pipe, schedule_type, schedule_prediction_type):

    if "Flux" in str(pipe.__class__.__name__):
        return None

    # Get the configuration for the selected schedule
    selected_schedule = SCHEDULE_TYPES.get(schedule_type)

    if selected_schedule:
        # Set all schedule types to False first
        default_config = {
            "use_karras_sigmas": False,
            "use_exponential_sigmas": False,
            "use_beta_sigmas": False,
        }
        pipe.scheduler.register_to_config(**default_config)

        # Apply the specific configuration for the selected schedule
        pipe.scheduler.register_to_config(**selected_schedule)

    # Get the configuration for the selected prediction type
    selected_prediction = SCHEDULE_PREDICTION_TYPE.get(
        schedule_prediction_type
    )

    if selected_prediction:
        # Update the prediction type in the scheduler's config
        if isinstance(selected_prediction, dict):
            pipe.scheduler.register_to_config(**selected_prediction)
        else:
            pipe.scheduler.register_to_config(
                prediction_type=selected_prediction
            )


def verify_schedule_integrity(model_scheduler):
    if (
        hasattr(model_scheduler.config, "prediction_type")
        and model_scheduler.config.prediction_type == "v_prediction"
    ):
        model_scheduler.register_to_config(
            rescale_betas_zero_snr=True,
        )

    if not hasattr(model_scheduler.config, "algorithm_type"):
        return model_scheduler

    logger.debug("Resetting scheduler settings")

    scheduler_xl = hf_hub_download(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        filename="scheduler/scheduler_config.json"
    )
    with open(scheduler_xl, 'r', encoding="utf-8") as file:
        params_ = json.load(file)

    original_scheduler = EulerDiscreteScheduler.from_config(params_)

    model_params_ = dict(model_scheduler.config.items())
    original_params_ = dict(original_scheduler.config.items())

    new_value_params = {}
    for k, v in model_params_.items():
        if not k.startswith("_") and k in original_params_:
            new_value_params[k] = v

    logger.debug(
        "The next configurations are loaded"
        f" from the repo model scheduler: {(new_value_params)}"
    )

    original_scheduler.register_to_config(
        **new_value_params
    )

    if (
        hasattr(original_scheduler.config, "prediction_type")
        and original_scheduler.config.prediction_type == "v_prediction"
    ):
        original_scheduler.register_to_config(
            rescale_betas_zero_snr=True,
        )

    return original_scheduler


def check_scheduler_compatibility(sampler, schedule_type):
    incompatible_schedule = INCOMPATIBILITY_SAMPLER_SCHEDULE.get(sampler, [])
    if schedule_type in incompatible_schedule:
        logger.warning(
            f"The sampler: {sampler} with schedule type: {schedule_type} "
            "is not well supported; changed to 'Automatic'."
        )
        schedule_type = "Automatic"

    return schedule_type


def ays_timesteps(cls, schedule):
    if schedule not in AYS_SCHEDULES:
        return {}

    list_steps = AYS_SCHEDULES[schedule]

    if cls == "StableDiffusionPipeline":
        steps = list_steps[0]
    elif cls == "StableDiffusionXLPipeline":
        steps = list_steps[1]
    else:
        raise ValueError(
            f"The pipeline {cls} does not support AYS scheduling."
        )

    key_param = "sigmas" if "beta" in schedule else "timesteps"

    return {key_param: steps}
