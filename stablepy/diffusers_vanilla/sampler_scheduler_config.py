from diffusers import EulerDiscreteScheduler
import json
from huggingface_hub import hf_hub_download
from ..logging.logging_setup import logger
from .constants import (
    SD15,
    SDXL,
    FLUX,
    INCOMPATIBILITY_SAMPLER_SCHEDULE,
    SCHEDULE_TYPES,
    SCHEDULE_TYPE_OPTIONS,
    SCHEDULE_PREDICTION_TYPE,
    AYS_SCHEDULES,
    FLUX_SCHEDULE_TYPES,
    FLUX_SCHEDULE_TYPE_OPTIONS,
)
import numpy as np


def configure_scheduler(pipe, schedule_type, schedule_prediction_type):

    if "Flux" in str(pipe.__class__.__name__):

        flux_selected_schedule = FLUX_SCHEDULE_TYPES.get(schedule_type, None)
        if flux_selected_schedule:
            pipe.scheduler.register_to_config(**flux_selected_schedule)

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

    if (
        hasattr(pipe.scheduler.config, "prediction_type")
        and pipe.scheduler.config.prediction_type == "v_prediction"
    ):
        pipe.scheduler.register_to_config(
            rescale_betas_zero_snr=True,
        )


def verify_schedule_integrity(model_scheduler, base_model_id):
    # noobai v-pred repo id
    if base_model_id.lower().startswith("laxhar/noobai-xl-vpred-"):
        model_scheduler.register_to_config(
            prediction_type="v_prediction",
        )

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


def check_scheduler_compatibility(cls, sampler, schedule_type):
    msg = ""
    auto_schedule = SCHEDULE_TYPE_OPTIONS[0]

    for old_sampler_config, def_schedule in [("Karras", "Karras"), ("trailing", "SGM Uniform"), (" Lu", "Lambdas")]:
        if old_sampler_config in sampler:
            sampler = sampler.replace(old_sampler_config, "").strip()
            if schedule_type == auto_schedule:
                schedule_type = def_schedule

    if cls == FLUX:
        if "Flow" not in sampler:
            sampler = "FlowMatch DPM++ 2M"
            msg += (
                "The selected sampler does not work with FLUX models;"
                f" so it has been switched to {sampler}. "
            )

        valid_schedule = FLUX_SCHEDULE_TYPES.get(schedule_type, None)

        if schedule_type != auto_schedule:
            if sampler == "FlowMatch Euler":
                msg += (
                    "FlowMatch Euler only support"
                    f" '{auto_schedule}' schedule type."
                )
                schedule_type = auto_schedule
            elif not valid_schedule:
                msg += (
                    f"The sampler: {sampler} only support schedule types"
                    f": {', '.join(FLUX_SCHEDULE_TYPE_OPTIONS)}"
                    f". Changed to '{auto_schedule}'."
                )
                schedule_type = auto_schedule

        return sampler, schedule_type, msg

    if "Flow" in sampler:
        sampler = sampler.replace("FlowMatch ", "")
        msg += (
            "The selected sampler works only with FLUX models;"
            f" so it has been switched to {sampler}. "
        )

    incompatible_schedule = INCOMPATIBILITY_SAMPLER_SCHEDULE.get(sampler, [])
    if schedule_type in incompatible_schedule:
        COMPATIBLE_SCHEDULES = [
            item for item in SCHEDULE_TYPE_OPTIONS
            if item not in incompatible_schedule
        ]

        msg += (
            f"The sampler: {sampler} only support schedule types"
            f": {', '.join(COMPATIBLE_SCHEDULES)}"
            f". Changed to '{auto_schedule}'."
        )
        schedule_type = auto_schedule

    return sampler, schedule_type, msg


def loglinear_interp(t_steps, num_steps):
    """
    Performs log-linear interpolation of a given array of decreasing numbers.
    """
    xs = np.linspace(0, 1, len(t_steps))
    ys = np.log(t_steps[::-1])

    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)

    interped_ys = np.exp(new_ys)[::-1].copy()
    return interped_ys


def ays_timesteps(cls, schedule, num_steps):
    if schedule not in AYS_SCHEDULES:
        return {}

    list_steps = AYS_SCHEDULES[schedule]

    if cls == SD15:
        steps = list_steps[0]
    elif cls == SDXL:
        steps = list_steps[1]
    else:
        raise ValueError(
            f"The pipeline {cls} does not support AYS scheduling."
        )

    key_param = "sigmas" if "sigmas" in schedule else "timesteps"

    if schedule == "AYS timesteps":
        t_steps = loglinear_interp(steps, num_steps)
        # steps = t_steps[t_steps != 0]
        # t_steps = np.append(t_steps, 0)
        steps = np.round(t_steps).astype(int).tolist()
    elif schedule == "AYS sigmas":
        t_steps = loglinear_interp(steps, num_steps)
        t_steps[-1] = .0
        steps = t_steps.tolist()

    return {key_param: steps}
