from __future__ import annotations
import math
from transformers import CLIPTextModelWithProjection
from collections import namedtuple
import torch
import gc
import re
import random

re_attention = re.compile(r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:\s*([+-]?[.\d]+)\s*\)|
\)|
]|
[^\\()\[\]:]+|
:
""", re.X)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

_EPS = 1e-12


def parse_prompt_attention(text):
    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith('\\'):
            res.append([text[1:], 1.0])
        elif text == '(':
            round_brackets.append(len(res))
        elif text == '[':
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ')' and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == ']' and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def _prepare_multipliers(self):
    # Ensure multipliers are a float tensor on same device/dtype as z and shaped (...,1)
    m = self.multipliers
    if not isinstance(m, torch.Tensor):
        m = torch.as_tensor(m, dtype=self.z.dtype, device=self.z.device)
    else:
        m = m.to(dtype=self.z.dtype, device=self.z.device)
    return m.reshape(m.shape + (1,))


class Emphasis:
    """Emphasis class decides how to death with (emphasized:1.1) text in prompts"""

    name: str = "Base"
    description: str = ""

    tokens: list[list[int]]
    """tokens from the chunk of the prompt"""

    multipliers: torch.Tensor
    """tensor with multipliers, once for each token"""

    z: torch.Tensor
    """output of cond transformers network (CLIP)"""

    def after_transformers(self):
        """Called after cond transformers network has processed the chunk of the prompt; this function should modify self.z to apply the emphasis"""

        pass


class EmphasisNone(Emphasis):
    name = "None"
    description = "disable the mechanism entirely and treat (:.1.1) as literal characters"


class EmphasisIgnore(Emphasis):
    name = "Ignore"
    description = "treat all empasised words as if they have no emphasis"


class EmphasisOriginal(Emphasis):
    name = "Original"
    description = "the original emphasis implementation"

    def after_transformers(self):
        original_mean = self.z.mean()
        self.z = self.z * self.multipliers.reshape(self.multipliers.shape + (1,)).expand(self.z.shape)

        # restoring original mean is likely not correct, but it seems to work well to prevent artifacts that happen otherwise
        new_mean = self.z.mean()
        self.z = self.z * (original_mean / new_mean)


class EmphasisOriginalNoNorm(EmphasisOriginal):
    name = "No norm"
    description = "same as original, but without normalization (seems to work better for SDXL)"

    def after_transformers(self):
        self.z = self.z * self.multipliers.reshape(self.multipliers.shape + (1,)).expand(self.z.shape)


class EmphasisMeanNorm(Emphasis):
    name = "MeanNorm"
    description = "Multiply token embeddings by multipliers and renormalize to preserve original mean."

    def after_transformers(self):
        m = _prepare_multipliers(self)
        original_mean = float(self.z.mean().detach().cpu().item())
        self.z = self.z * m.expand_as(self.z)
        new_mean = float(self.z.mean().detach().cpu().item())
        if abs(new_mean) > _EPS:
            self.z = self.z * (original_mean / new_mean)


class EmphasisL2Norm(Emphasis):
    name = "L2Norm"
    description = "Favor positive (>1) multipliers strongly, downmagnitude <1 tokens; preserve overall norm."

    def after_transformers(self):
        m = _prepare_multipliers(self)
        base = self.z

        # Heuristic post-transformer approximation of 'compel':
        # - square >1 multipliers to emphasize positives,
        # - keep <1 multipliers as-is so negatives are still down-magnitude,
        # - then re-scale to preserve overall L2 norm.
        f = torch.where(m > 1.0, m ** 2, m)  # shape (B, T, 1)
        weighted = base * f.expand_as(base)

        base_norm = float(torch.linalg.norm(base).detach().cpu().item())
        weighted_norm = float(torch.linalg.norm(weighted).detach().cpu().item())
        if weighted_norm > _EPS:
            weighted = weighted * (base_norm / weighted_norm)

        self.z = weighted


class EmphasisDownMagnitudeL2Norm(Emphasis):
    name = "DownMagnitude"
    description = "Down-magnitude: scale tokens with multipliers < 1 while preserving total embedding magnitude."

    def after_transformers(self):
        m = _prepare_multipliers(self)
        base = self.z

        # tokens <1 keep m, tokens >=1 become 1 (no up-weight)
        down_m = torch.where(m < 1.0, m, torch.ones_like(m))
        weighted = base * down_m.expand_as(base)

        base_norm = float(torch.linalg.norm(base).detach().cpu().item())
        weighted_norm = float(torch.linalg.norm(weighted).detach().cpu().item())
        if weighted_norm > _EPS:
            weighted = weighted * (base_norm / weighted_norm)

        self.z = weighted


class EmphasisMeanNormDownMagnitude(Emphasis):
    name = "MeanNormDownMagnitude"
    description = "Hybrid — multiply + an additional downmagnitude-like correction for tokens <1."

    def after_transformers(self):
        m = _prepare_multipliers(self)
        base = self.z

        # multiply and renormalize to preserve mean
        interpolate = base * m.expand_as(base)
        interpolate_mean = float(interpolate.mean().detach().cpu().item())
        base_mean = float(base.mean().detach().cpu().item())
        if abs(interpolate_mean) > _EPS:
            interpolate = interpolate * (base_mean / interpolate_mean)

        # downmagnitude correction: tokens with m < 1 get the direct multiplicative effect
        down_mask = (m < 1.0).to(dtype=self.z.dtype)
        down_effect = base * (1 - (1 - m) * down_mask).expand_as(base)

        out = interpolate * 0.8 + down_effect * 0.2

        base_norm = float(torch.linalg.norm(base).detach().cpu().item())
        out_norm = float(torch.linalg.norm(out).detach().cpu().item())
        if out_norm > _EPS:
            out = out * (base_norm / out_norm)

        self.z = out


class EmphasisExponentialScale(Emphasis):
    name = "ExponentialScale"
    description = "Apply multipliers with exponential scaling"

    def after_transformers(self):
        m = _prepare_multipliers(self)
        base = self.z
        f = torch.exp(m - 1.0)  # m=1 → 1.0, m=1.5 → ~1.65
        self.z = base * f.expand_as(base)


class EmphasisSigmoidScale(Emphasis):
    name = "SigmoidScale"
    description = "Squash multipliers with sigmoid-like curve to prevent extremes"

    def after_transformers(self):
        m = _prepare_multipliers(self)
        f = 1 + torch.tanh(m - 1.0)  # maps (-inf,+inf) → (0,2)
        self.z = self.z * f.expand_as(self.z)


class EmphasisContrastiveShift(Emphasis):
    name = "ContrastiveShift"
    description = "Boost emphasized tokens, reduce non-emphasized ones"

    def after_transformers(self):
        m = _prepare_multipliers(self)
        base = self.z
        mean_m = m.mean(dim=1, keepdim=True)  # per-sequence average
        f = m - mean_m + 1.0  # shift relative to avg
        self.z = base * f.expand_as(base)


class EmphasisStochasticNoise(Emphasis):
    name = "StochasticNoise"
    description = "Adds small random noise to multipliers for variety"

    def after_transformers(self):
        m = _prepare_multipliers(self)
        noise = (torch.randn_like(m) * 0.05)  # ±5% noise
        f = m + noise
        self.z = self.z * f.expand_as(self.z)


class EmphasisMeanNormExpoBlend(Emphasis):
    name = "MeanNormExpoBlend"
    description = "Blend of interpolate and exponential emphasis"

    def after_transformers(self):
        m = _prepare_multipliers(self)
        interpolate = self.z * m.expand_as(self.z)
        expo = self.z * torch.exp(m - 1.0).expand_as(self.z)
        self.z = 0.5 * interpolate + 0.5 * expo


class EmphasisExponentialGammaNorm(Emphasis):
    name = "ExponentialGammaNorm"
    description = "Exponentially scale multipliers (m^γ). Strongly amplifies both >1 and <1."

    gamma = 1.30  # tunable strength

    def after_transformers(self):
        m = _prepare_multipliers(self)
        weighted = self.z * (m ** self.gamma).expand_as(self.z)
        # norm-preserve
        base_norm = float(torch.linalg.norm(self.z).detach().cpu().item())
        new_norm = float(torch.linalg.norm(weighted).detach().cpu().item())
        if new_norm > _EPS:
            weighted = weighted * (base_norm / new_norm)
        self.z = weighted


class EmphasisWaveModulation(Emphasis):
    name = "WaveModulation"
    description = "Apply a sinusoidal modulation to the embeddings based on multipliers."

    def after_transformers(self):
        m = _prepare_multipliers(self)
        # Sine oscillation with multiplier as frequency
        phase = torch.arange(self.z.shape[1], device=self.z.device).float()
        phase = phase.unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
        wave = torch.sin(phase * m)
        weighted = self.z * (1 + 0.5 * wave).expand_as(self.z)
        self.z = weighted


class EmphasisContrastAlpha(Emphasis):
    name = "ContrastAlpha"
    description = "Boost separation between emphasized and neutral tokens."

    alpha = 1.1  # contrast factor

    def after_transformers(self):
        m = _prepare_multipliers(self)
        centered = self.z - self.z.mean(dim=1, keepdim=True)
        weighted = centered * m.expand_as(self.z) * self.alpha
        self.z = weighted + self.z.mean(dim=1, keepdim=True)


class EmphasisGaussianNoise(Emphasis):
    name = "Noisy"
    description = "Adds controlled Gaussian noise based on multiplier intensity."

    noise_scale = 0.1

    def after_transformers(self):
        m = _prepare_multipliers(self)
        noise = torch.randn_like(self.z) * self.noise_scale * (m - 1.0).abs()
        self.z = self.z * m.expand_as(self.z) + noise


class EmphasisGravityAttract(Emphasis):
    name = "GravityAttract"
    description = "Tokens with higher multipliers attract nearby tokens."

    def after_transformers(self):
        m = _prepare_multipliers(self)
        weights = m / (m.sum(dim=1, keepdim=True) + _EPS)
        gravity_center = (self.z * weights.expand_as(self.z)).sum(dim=1, keepdim=True)
        self.z = self.z + (gravity_center - self.z) * m.expand_as(self.z) * 0.2


class EmphasisRandomMix(Emphasis):
    name = "RandomMix"
    description = "Randomly mix multiple emphasis strategies for creative variety."
    gamma = 1.25
    alpha = 1.1
    noise_scale = 0.1

    def after_transformers(self):
        choice = random.choice([0, 1, 2])
        if choice == 0:
            EmphasisExponentialGammaNorm.after_transformers(self)
        elif choice == 1:
            EmphasisContrastAlpha.after_transformers(self)
        else:
            EmphasisGaussianNoise.after_transformers(self)


def recover_dist(base_emb, weighted_emb):
    fixed_std = (base_emb.std() / weighted_emb.std()) * (weighted_emb - weighted_emb.mean())
    embeddings_final = fixed_std + (base_emb.mean() - fixed_std.mean())
    return embeddings_final


class EmphasisInterpolate(Emphasis):
    name = "Interpolate"  # MeanNorm2
    description = "Interpolate-style: interpolate embeddings between empty prompt and original."

    def after_transformers(self):
        m = _prepare_multipliers(self)                # multipliers per token
        base = self.z
        weighted = base * m.expand_as(base)           # scale relative to empty
        # recover mean/std distribution like interpolate does
        self.z = recover_dist(base, weighted)


class EmphasisScaledNormalize(Emphasis):
    name = "ScaledNormalize"
    description = "Scale embeddings by normalized multipliers and restore mean/std distribution."

    def after_transformers(self):
        m = _prepare_multipliers(self)
        base = self.z
        top = m.max()
        w_max = 1.0
        scale = (m / top) * w_max
        weighted = base * scale.expand_as(base)
        self.z = recover_dist(base, weighted)


def get_current_option(emphasis_option_name):
    return next(iter([x for x in options if x.name == emphasis_option_name]), EmphasisOriginal)


def get_options_descriptions():
    return ", ".join(f"{x.name}: {x.description}" for x in options)


options = [
    EmphasisNone,
    EmphasisIgnore,
    EmphasisOriginal,
    EmphasisOriginalNoNorm,
    EmphasisMeanNorm,
    EmphasisL2Norm,
    EmphasisMeanNormDownMagnitude,
    EmphasisExponentialScale,
    EmphasisSigmoidScale,
    EmphasisContrastiveShift,
    EmphasisStochasticNoise,
    EmphasisMeanNormExpoBlend,
    EmphasisExponentialGammaNorm,
    EmphasisWaveModulation,
    EmphasisContrastAlpha,
    EmphasisGaussianNoise,
    EmphasisGravityAttract,
    EmphasisRandomMix,
    EmphasisDownMagnitudeL2Norm,
    EmphasisInterpolate,
    EmphasisScaledNormalize,
]


class PromptChunk:
    """
    This object contains token ids, weight (multipliers:1.4) and textual inversion embedding info for a chunk of prompt.
    If a prompt is short, it is represented by one PromptChunk, otherwise, multiple are necessary.
    Each PromptChunk contains an exact amount of tokens - 77, which includes one for start and end token,
    so just 75 tokens from prompt.
    """

    def __init__(self):
        self.tokens = []
        self.multipliers = []
        self.fixes = []


PromptChunkFix = namedtuple('PromptChunkFix', ['offset', 'embedding'])
"""An object of this type is a marker showing that textual inversion embedding's vectors have to placed at offset in the prompt
chunk. Those objects are found in PromptChunk.fixes and, are placed into FrozenCLIPEmbedderWithCustomWordsBase.hijack.fixes, and finally
are applied by sd_hijack.EmbeddingsWithFixes's forward function."""


class ClassicTextProcessingEngine:
    def __init__(
            self,
            text_encoder,
            tokenizer,
            chunk_length=75,
            emphasis_name="Original",
            text_projection=False,
            minimal_clip_skip=1,
            clip_skip=1,
            return_pooled=False,
            final_layer_norm=True,
    ):
        super().__init__()

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

        self.device = text_encoder.device.type
        self.emphasis = get_current_option(emphasis_name)()
        self.text_projection = text_projection
        self.map_to_projected_space = None

        if self.text_projection:
            if isinstance(self.text_encoder, CLIPTextModelWithProjection):
                self.map_to_projected_space = self.text_encoder.text_projection
            else:
                embed_dim = self.text_encoder.config.hidden_size
                self.map_to_projected_space = torch.nn.Linear(
                    embed_dim, embed_dim, bias=False
                ).to(self.device, dtype=self.text_encoder.dtype)

        self.minimal_clip_skip = minimal_clip_skip
        self.clip_skip = clip_skip
        self.return_pooled = return_pooled
        self.final_layer_norm = final_layer_norm

        self.chunk_length = chunk_length

        self.id_start = self.tokenizer.bos_token_id
        self.id_end = self.tokenizer.eos_token_id
        self.id_pad = self.tokenizer.pad_token_id

        vocab = self.tokenizer.get_vocab()

        self.comma_token = vocab.get(',</w>', None)

        self.token_mults = {}

        tokens_with_parens = [(k, v) for k, v in vocab.items() if '(' in k or ')' in k or '[' in k or ']' in k]
        for text, ident in tokens_with_parens:
            mult = 1.0
            for c in text:
                if c == '[':
                    mult /= 1.1
                if c == ']':
                    mult *= 1.1
                if c == '(':
                    mult *= 1.1
                if c == ')':
                    mult /= 1.1

            if mult != 1.0:
                self.token_mults[ident] = mult

    def empty_chunk(self):
        """creates an empty PromptChunk and returns it"""

        chunk = PromptChunk()
        chunk.tokens = [self.id_start] + [self.id_end] * (self.chunk_length + 1)
        chunk.multipliers = [1.0] * (self.chunk_length + 2)
        return chunk

    def get_target_prompt_token_count(self, token_count):
        """returns the maximum number of tokens a prompt of a known length can have before it requires one more PromptChunk to be represented"""

        return math.ceil(max(token_count, 1) / self.chunk_length) * self.chunk_length

    def tokenize(self, texts):
        """Converts a batch of texts into a batch of token ids"""
        tokenized = self.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]

        return tokenized

    def encode_with_transformers(self, tokens):
        """
        converts a batch of token ids (in python lists) into a single tensor with numeric representation of those tokens;
        All python lists with tokens are assumed to have same length, usually 77.
        if input is a list with B elements and each element has T tokens, expected output shape is (B, T, C), where C depends on
        model - can be 768 and 1024.
        Among other things, this call will read self.hijack.fixes, apply it to its inputs, and clear it (setting it to None).
        """

        tokens = tokens.to(self.device)

        outputs = self.text_encoder.text_model(input_ids=tokens, output_hidden_states=True)

        layer_id = - max(self.clip_skip, self.minimal_clip_skip)
        z = outputs.hidden_states[layer_id]

        if self.final_layer_norm:
            z = self.text_encoder.text_model.final_layer_norm(z)

        if self.return_pooled:
            pooled_output = outputs.pooler_output

            if self.text_projection:
                pooled_output = self.map_to_projected_space(pooled_output)

            z.pooled = pooled_output
        return z

    def tokenize_line(self, line):

        if self.emphasis.name != "None":
            parsed = parse_prompt_attention(line)
        else:
            parsed = [[line, 1.0]]

        tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = PromptChunk()
        token_count = 0
        last_comma = -1

        def next_chunk(is_last=False):
            """puts current chunk into the list of results and produces the next one - empty;
            if is_last is true, tokens <end-of-text> tokens at the end won't add to token_count"""
            nonlocal token_count
            nonlocal last_comma
            nonlocal chunk

            if is_last:
                token_count += len(chunk.tokens)
            else:
                token_count += self.chunk_length

            to_add = self.chunk_length - len(chunk.tokens)
            if to_add > 0:
                chunk.tokens += [self.id_end] * to_add
                chunk.multipliers += [1.0] * to_add

            chunk.tokens = [self.id_start] + chunk.tokens + [self.id_end]
            chunk.multipliers = [1.0] + chunk.multipliers + [1.0]

            last_comma = -1
            chunks.append(chunk)
            chunk = PromptChunk()

        for tokens, (text, weight) in zip(tokenized, parsed):
            if text == 'BREAK' and weight == -1:
                next_chunk()
                continue

            position = 0
            while position < len(tokens):
                token = tokens[position]

                comma_padding_backtrack = 20

                if token == self.comma_token:
                    last_comma = len(chunk.tokens)

                # this is when we are at the end of allotted 75 tokens for the current chunk, and the current token is not a comma. opts.comma_padding_backtrack
                # is a setting that specifies that if there is a comma nearby, the text after the comma should be moved out of this chunk and into the next.
                elif comma_padding_backtrack != 0 and len(chunk.tokens) == self.chunk_length and last_comma != -1 and len(chunk.tokens) - last_comma <= comma_padding_backtrack:
                    break_location = last_comma + 1

                    reloc_tokens = chunk.tokens[break_location:]
                    reloc_mults = chunk.multipliers[break_location:]

                    chunk.tokens = chunk.tokens[:break_location]
                    chunk.multipliers = chunk.multipliers[:break_location]

                    next_chunk()
                    chunk.tokens = reloc_tokens
                    chunk.multipliers = reloc_mults

                if len(chunk.tokens) == self.chunk_length:
                    next_chunk()

                chunk.tokens.append(token)
                chunk.multipliers.append(weight)
                position += 1

        if chunk.tokens or not chunks:
            next_chunk(is_last=True)

        return chunks, token_count

    def process_texts(self, texts):

        token_count = 0

        cache = {}
        batch_chunks = []
        for line in texts:
            if line in cache:
                chunks = cache[line]
            else:
                chunks, current_token_count = self.tokenize_line(line)
                token_count = max(current_token_count, token_count)

                cache[line] = chunks

            batch_chunks.append(chunks)

        return batch_chunks, token_count

    def __call__(self, texts):
        if not isinstance(texts, list):
            texts = [texts]

        batch_chunks, token_count = self.process_texts(texts)

        chunk_count = max([len(x) for x in batch_chunks])

        zs = []
        for i in range(chunk_count):
            batch_chunk = [chunks[i] if i < len(chunks) else self.empty_chunk() for chunks in batch_chunks]

            tokens = [x.tokens for x in batch_chunk]
            multipliers = [x.multipliers for x in batch_chunk]

            z = self.process_tokens(tokens, multipliers)
            zs.append(z)

        if self.return_pooled:
            return torch.hstack(zs), zs[0].pooled
        else:
            return torch.hstack(zs)

    def process_tokens(self, remade_batch_tokens, batch_multipliers):
        tokens = torch.asarray(remade_batch_tokens)

        # this is for SD2: SD1 uses the same token for padding and end of text, while SD2 uses different ones.
        if self.id_end != self.id_pad:
            for batch_pos in range(len(remade_batch_tokens)):
                index = remade_batch_tokens[batch_pos].index(self.id_end)
                tokens[batch_pos, index + 1:tokens.shape[1]] = self.id_pad

        z = self.encode_with_transformers(tokens)

        pooled = getattr(z, 'pooled', None)

        self.emphasis.tokens = remade_batch_tokens
        self.emphasis.multipliers = torch.asarray(batch_multipliers).to(z)
        self.emphasis.z = z
        self.emphasis.after_transformers()
        z = self.emphasis.z

        if pooled is not None:
            z.pooled = pooled

        return z


def pad_equal_len(text_embedder, cond_embeddings, uncond_embeddings):

    cond_len = cond_embeddings.shape[1]
    uncond_len = uncond_embeddings.shape[1]

    if cond_len == uncond_len:
        all_embeddings = [cond_embeddings, uncond_embeddings]
    else:
        if cond_len > uncond_len:
            n = (cond_len - uncond_len) // 77
            all_embeddings = [cond_embeddings, torch.cat([uncond_embeddings] + [text_embedder([""])] * n, dim=1)]
        else:
            n = (uncond_len - cond_len) // 77
            all_embeddings = [torch.cat([cond_embeddings] + [text_embedder([""])] * n, dim=1), uncond_embeddings]

    return all_embeddings


def text_embeddings_equal_len(text_embedder, prompt, negative_prompt, get_pooled=False):
    pooled_cond = pooled_neg_cond = None

    cond_embeddings = text_embedder(prompt, get_pooled)
    uncond_embeddings = text_embedder(negative_prompt, get_pooled)

    if isinstance(cond_embeddings, tuple):
        cond_embeddings, pooled_cond = cond_embeddings[0], cond_embeddings[1]
        uncond_embeddings, pooled_neg_cond = uncond_embeddings[0], uncond_embeddings[1]

    cond_len = cond_embeddings.shape[1]
    uncond_len = uncond_embeddings.shape[1]

    if cond_len == uncond_len:
        all_embeddings = [cond_embeddings, uncond_embeddings]
    else:
        if cond_len > uncond_len:
            n = (cond_len - uncond_len) // 77
            all_embeddings = [cond_embeddings, torch.cat([uncond_embeddings] + [text_embedder([""])] * n, dim=1)]
        else:
            n = (uncond_len - cond_len) // 77
            all_embeddings = [torch.cat([cond_embeddings] + [text_embedder([""])] * n, dim=1), uncond_embeddings]

    if get_pooled:
        return all_embeddings + [pooled_cond, pooled_neg_cond]
    else:
        return all_embeddings
