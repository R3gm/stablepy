from .multi_emphasis_prompt import (
    StableDiffusionLongPromptProcessor,
    text_embeddings_equal_len,
)
import torch
import gc
from compel import Compel, ReturnedEmbeddingsType
from ..logging.logging_setup import logger
from .constants import (
    PROMPT_WEIGHT_OPTIONS,
    OLD_PROMPT_WEIGHT_OPTIONS,
    SD_EMBED,
    CLASSIC_VARIANT,
    ALL_PROMPT_WEIGHT_OPTIONS,
)
from .prompt_weights import get_embed_new
from .sd_embed.embedding_funcs import (
    get_weighted_text_embeddings_sd15,
    get_weighted_text_embeddings_sdxl,
    get_weighted_text_embeddings_flux1,
)


class Prompt_Embedder_Base:
    def __init__(self):
        self.last_clip_skip = None

    def apply_ti(self, class_name, textual_inversion, pipe, device, gui_active):

        if "FluxPipeline" == class_name:
            logger.warning("Textual Inverstion not available")
            return None

        # Textual Inversion
        for name, directory_name in textual_inversion:
            try:
                if class_name == "StableDiffusionPipeline":
                    if directory_name.endswith(".pt"):
                        model = torch.load(directory_name, map_location=device)
                        model_tensors = model.get("string_to_param").get("*")
                        s_model = {"emb_params": model_tensors}
                        # save_file(s_model, directory_name[:-3] + '.safetensors')
                        pipe.load_textual_inversion(s_model, token=name)
                    else:
                        # pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer),pad_to_multiple_of=128)
                        # pipe.load_textual_inversion("./bad_prompt.pt", token="baddd")
                        pipe.load_textual_inversion(directory_name, token=name)
                elif class_name == "StableDiffusionXLPipeline":
                    from safetensors.torch import load_file
                    state_dict = load_file(directory_name)
                    pipe.load_textual_inversion(state_dict["clip_g"], token=name, text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
                    pipe.load_textual_inversion(state_dict["clip_l"], token=name, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
                else:
                    logger.error("Textual Inversion not combatible")

                logger.info(f"Applied: {name}")
            except Exception as e:
                exception = str(e)
                if name in exception:
                    logger.debug(f"Previous loaded embed {name}")
                else:
                    logger.error(exception)
                    logger.error(f"Can't apply embed {name}")

    @torch.no_grad()
    def __call__(self, prompt, negative_prompt, syntax_weights, pipe, clip_skip, compel):
        if syntax_weights in CLASSIC_VARIANT:
            emphasis = CLASSIC_VARIANT[syntax_weights]
            return self.classic_variant(
                prompt, negative_prompt, pipe, clip_skip, emphasis
            )
        elif syntax_weights in SD_EMBED:
            return self.sd_embed_variant(
                prompt, negative_prompt, pipe, clip_skip
            )
        else:
            return self.compel_processor(
                prompt, negative_prompt, pipe, clip_skip, syntax_weights, compel
            )


class Promt_Embedder_SD1(Prompt_Embedder_Base):

    def classic_variant(self, prompt, negative_prompt, pipe, clip_skip, emphasis):
        text_embedder = StableDiffusionLongPromptProcessor(
            pipe,
            pipe.tokenizer,
            pipe.text_encoder,
            2 if clip_skip else 1,
            emphasis,
            20,
        )

        cond_embeddings, uncond_embeddings = text_embeddings_equal_len(text_embedder, prompt, negative_prompt)
        return cond_embeddings, uncond_embeddings, None

    def sd_embed_variant(self, prompt, negative_prompt, pipe, clip_skip):

        (
            cond_embeddings,
            uncond_embeddings
        ) = get_weighted_text_embeddings_sd15(
            pipe,
            prompt=prompt,
            neg_prompt=negative_prompt,
        )

        return cond_embeddings, uncond_embeddings, None

    def compel_processor(self, prompt, negative_prompt, pipe, clip_skip, syntax_weights, compel):

        if compel is None or clip_skip != self.last_clip_skip:
            compel = Compel(
                tokenizer=pipe.tokenizer,
                text_encoder=pipe.text_encoder,
                truncate_long_prompts=False,
                returned_embeddings_type=(
                    ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED
                    if clip_skip
                    else ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED),
            )
            self.last_clip_skip = clip_skip

        # Syntax weights
        compel_weights = False if syntax_weights == "Classic" else True
        # pipe.to(device)
        prompt_emb = get_embed_new(
            prompt, pipe, compel, compel_process_sd=compel_weights
        )
        negative_prompt_emb = get_embed_new(
            negative_prompt, pipe, compel, compel_process_sd=compel_weights
        )

        # Fix error shape
        if prompt_emb.shape != negative_prompt_emb.shape:
            (
                prompt_emb,
                negative_prompt_emb,
            ) = compel.pad_conditioning_tensors_to_same_length(
                [prompt_emb, negative_prompt_emb]
            )

        return prompt_emb, negative_prompt_emb, compel


class Promt_Embedder_SDXL(Prompt_Embedder_Base):
    def classic_variant(self, prompt, negative_prompt, pipe, clip_skip, emphasis):

        text_embedder = StableDiffusionLongPromptProcessor(
            pipe,
            pipe.tokenizer,
            pipe.text_encoder,
            2 if clip_skip else 1,  # disabled with sdxl
            emphasis,
            20,
        )

        text_embedder_2 = StableDiffusionLongPromptProcessor(
            pipe,
            pipe.tokenizer_2,
            pipe.text_encoder_2,
            2 if clip_skip else 1,  # disabled with sdxl
            emphasis,
            20,
        )

        cond_embeddings, uncond_embeddings = text_embeddings_equal_len(text_embedder, prompt, negative_prompt, get_pooled=False)

        (
            cond_embeddings_2,
            uncond_embeddings_2,
            cond_pooled,
            uncond_pooled
        ) = text_embeddings_equal_len(text_embedder_2, prompt, negative_prompt, get_pooled=True)

        cond_embed = torch.cat((cond_embeddings, cond_embeddings_2), dim=2)
        neg_uncond_embed = torch.cat((uncond_embeddings, uncond_embeddings_2), dim=2)

        all_cond = torch.cat([cond_embed, neg_uncond_embed])

        all_pooled = torch.cat([cond_pooled, uncond_pooled])

        assert torch.equal(all_cond[0:1], cond_embed), "Tensors are not equal"

        return all_cond, all_pooled, None

    def sd_embed_variant(self, prompt, negative_prompt, pipe, clip_skip):

        (
            cond_embed,
            neg_uncond_embed,
            cond_pooled,
            uncond_pooled
        ) = get_weighted_text_embeddings_sdxl(
            pipe,
            prompt=prompt,
            neg_prompt=negative_prompt,
        )

        all_cond = torch.cat([cond_embed, neg_uncond_embed])

        all_pooled = torch.cat([cond_pooled, uncond_pooled])

        assert torch.equal(all_cond[0:1], cond_embed), "Tensors are not equal"

        return all_cond, all_pooled, None

    def compel_processor(self, prompt, negative_prompt, pipe, clip_skip, syntax_weights, compel):

        if compel is None or clip_skip != self.last_clip_skip:
            compel = Compel(
                tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                requires_pooled=[False, True],
                truncate_long_prompts=False,
                returned_embeddings_type=(
                    ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED
                    if clip_skip
                    else ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED
                ),
            )
            self.last_clip_skip = clip_skip

        # Syntax weights
        # pipe.to(device)
        if syntax_weights == "Classic":
            prompt = get_embed_new(
                prompt, pipe, compel, only_convert_string=True
            )
            negative_prompt = get_embed_new(
                negative_prompt, pipe, compel, only_convert_string=True
            )

        conditioning, pooled = compel([prompt, negative_prompt])

        return conditioning, pooled, compel


class Promt_Embedder_FLUX(Prompt_Embedder_Base):
    def classic_variant(self, prompt, negative_prompt, pipe, clip_skip, emphasis):

        # pipe.text_encoder_2.to("cuda")
        # pipe.transformer.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()

        text_embedder = StableDiffusionLongPromptProcessor(
            pipe,
            pipe.tokenizer,
            pipe.text_encoder,
            1,
            emphasis,
            20,
        )

        from .t5_embedder import T5TextProcessingEngine
        text_embedder_2 = T5TextProcessingEngine(
            pipe.text_encoder_2,
            pipe.tokenizer_2,
            emphasis_name=emphasis,
            min_length=(
                512 if pipe.transformer.config.guidance_embeds else 256
            ),
        )

        _, pooled_embeddings = text_embedder(prompt, get_pooled=True)
        positive_embeddings = text_embedder_2(prompt)

        pooled_embeddings = pooled_embeddings.to(dtype=pipe.text_encoder.dtype)
        positive_embeddings = positive_embeddings.to(dtype=pipe.text_encoder_2.dtype)

        # pipe.text_encoder_2.to("cpu")
        # pipe.transformer.to("cuda")
        torch.cuda.empty_cache()
        gc.collect()

        return positive_embeddings, pooled_embeddings, None

    def sd_embed_variant(self, prompt, negative_prompt, pipe, clip_skip):

        torch.cuda.empty_cache()
        gc.collect()

        (
            positive_embeddings,
            pooled_embeddings
        ) = get_weighted_text_embeddings_flux1(
            pipe=pipe,
            prompt=prompt,
        )

        positive_embeddings = positive_embeddings.to(dtype=pipe.text_encoder_2.dtype)
        pooled_embeddings = pooled_embeddings.to(dtype=pipe.text_encoder.dtype)

        torch.cuda.empty_cache()
        gc.collect()

        return positive_embeddings, pooled_embeddings, None

    def compel_processor(self, prompt, negative_prompt, pipe, clip_skip, syntax_weights, compel):

        # pipe.text_encoder_2.to("cuda")
        # pipe.transformer.to("cpu")
        # torch.cuda.empty_cache()
        # gc.collect()

        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=pipe.text_encoder.device,
            num_images_per_prompt=1,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            max_sequence_length=512,
            lora_scale=None,
        )

        # pipe.text_encoder_2.to("cpu")
        # pipe.transformer.to("cuda")
        torch.cuda.empty_cache()
        gc.collect()

        return prompt_embeds, pooled_prompt_embeds, None
