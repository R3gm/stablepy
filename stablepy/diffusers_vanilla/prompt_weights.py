# =====================================
# Prompt weights
# =====================================
from compel import Compel
from diffusers import StableDiffusionPipeline, DDIMScheduler
import re
from compel import ReturnedEmbeddingsType
import torch


def concat_tensor(t):
    t_list = torch.split(t, 1, dim=0)
    t = torch.cat(t_list, dim=1)
    return t


def merge_embeds(prompt_chanks, compel):
    num_chanks = len(prompt_chanks)
    power_prompt = 1 / (num_chanks * (num_chanks + 1) // 2)
    prompt_embs = compel(prompt_chanks)
    t_list = list(torch.split(prompt_embs, 1, dim=0))
    for i in range(num_chanks):
        t_list[-(i + 1)] = t_list[-(i + 1)] * ((i + 1) * power_prompt)
    prompt_emb = torch.stack(t_list, dim=0).sum(dim=0)
    return prompt_emb


def detokenize(chunk, actual_prompt):
    chunk[-1] = chunk[-1].replace("</w>", "")
    chanked_prompt = "".join(chunk).strip()
    while "</w>" in chanked_prompt:
        if actual_prompt[chanked_prompt.find("</w>")] == " ":
            chanked_prompt = chanked_prompt.replace("</w>", " ", 1)
        else:
            chanked_prompt = chanked_prompt.replace("</w>", "", 1)
    actual_prompt = actual_prompt.replace(chanked_prompt, "")
    return chanked_prompt.strip(), actual_prompt.strip()


def tokenize_line(line, tokenizer):  # split into chunks
    actual_prompt = line.lower().strip()
    if actual_prompt == "":
        actual_prompt = "worst quality"
    actual_tokens = tokenizer.tokenize(actual_prompt)
    max_tokens = tokenizer.model_max_length - 2
    comma_token = tokenizer.tokenize(",")[0]

    chunks = []
    chunk = []
    for item in actual_tokens:
        chunk.append(item)
        if len(chunk) == max_tokens:
            if chunk[-1] != comma_token:
                for i in range(max_tokens - 1, -1, -1):
                    if chunk[i] == comma_token:
                        actual_chunk, actual_prompt = detokenize(
                            chunk[: i + 1], actual_prompt
                        )
                        chunks.append(actual_chunk)
                        chunk = chunk[i + 1 :]
                        break
                else:
                    actual_chunk, actual_prompt = detokenize(chunk, actual_prompt)
                    chunks.append(actual_chunk)
                    chunk = []
            else:
                actual_chunk, actual_prompt = detokenize(chunk, actual_prompt)
                chunks.append(actual_chunk)
                chunk = []
    if chunk:
        actual_chunk, _ = detokenize(chunk, actual_prompt)
        chunks.append(actual_chunk)

    return chunks


def prompt_weight_conversor(input_string):
    # Convert prompt weights from a1... to comel

    # Find and replace instances of the colon format with the desired format
    converted_string = re.sub(r"\(([^:]+):([\d.]+)\)", r"(\1)\2", input_string)

    # Find and replace square brackets with round brackets and assign weight
    converted_string = re.sub(r"\[([^:\]]+)\]", r"(\1)0.909090909", converted_string)

    # Handle the general case of [x:number] and convert it to (x)0.9
    converted_string = re.sub(r"\[([^:]+):[\d.]+\]", r"(\1)0.9", converted_string)

    # Add a '+' sign after the closing parenthesis if no weight is specified
    modified_string = re.sub(r"\(([^)]+)\)(?![\d.])", r"(\1)+", converted_string)

    # double (())
    # modified_string = re.sub(r'\(\(([^)]+)\)\+\)', r'(\1)++', modified_string)

    # triple ((()))
    # modified_string = re.sub(r'\(\(([^)]+)\)\+\+\)', r'(\1)+++', modified_string)

    # print(modified_string)
    return modified_string
