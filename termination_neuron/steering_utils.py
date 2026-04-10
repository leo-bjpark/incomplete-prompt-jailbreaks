from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import COMMON_PROMPT

IPJ_DATASET_NAME = "leo-bjpark/incomplete-prompt-jailbreak"
IPJ_DATASET_SETTING = "completion_benchmark"
IPJ_DATASET_SPLIT = "train"
SUPPORTED_AUGMENT_TYPES = {"comma", "continue", "example"}


def load_model_and_tokenizer(model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def normalize_augment_type(augment_type: str) -> str:
    aliases = {
        "one_way_is": "continue",
        "for_example": "example",
        "onewayis": "continue",
    }
    normalized = aliases.get(augment_type, augment_type)
    if normalized not in SUPPORTED_AUGMENT_TYPES:
        raise ValueError(f"Invalid augment type: {augment_type}. Choose one of {sorted(SUPPORTED_AUGMENT_TYPES)}")
    return normalized


def load_ipj_questions(dataset_name: str, dataset_setting: str, dataset_split: str) -> list[str]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, dataset_setting, split=dataset_split)
    return list(dataset["prompt"])


def build_prompts_with_common_prompt(
    *,
    questions: list[str],
    common_prompt: str,
    augment_type: str | None = None,
    max_samples: int | None = None,
) -> list[str]:
    selected_questions = questions[:max_samples] if max_samples is not None else questions
    prompts = [f"{common_prompt}\n{question}" for question in selected_questions]
    if augment_type is None:
        return prompts
    return [_apply_augment(prompt, augment_type) for prompt in prompts]


def _apply_augment(prompt: str, augment_type: str) -> str:
    augment_type = normalize_augment_type(augment_type)
    augmented = prompt.replace("?", ",")
    if augment_type == "comma":
        return augmented
    if augment_type == "continue":
        return f"{augmented} One way is"
    return f"{augmented} For example,"


def load_ipj_prompts(
    *,
    dataset_name: str,
    dataset_setting: str,
    dataset_split: str,
    augment_type: str,
    max_samples: int | None = None,
) -> list[str]:
    questions = load_ipj_questions(dataset_name, dataset_setting, dataset_split)
    common_prompt = COMMON_PROMPT
    return build_prompts_with_common_prompt(
        questions=questions,
        common_prompt=common_prompt,
        augment_type=augment_type,
        max_samples=max_samples,
    )


@dataclass
class HiddenStateHook:
    neuron_indices: torch.Tensor | None = None
    magnitude_alpha: float = 1.0

    def __call__(self, module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if self.neuron_indices is None or len(self.neuron_indices) == 0:
            return
        if inputs[0].shape[1] == 1:
            output[:, :, self.neuron_indices] = output[:, :, self.neuron_indices] * self.magnitude_alpha


def get_llm_blocks(model: torch.nn.Module, model_name: str) -> torch.nn.ModuleList:
    if model_name == "gpt2":
        return model.transformer.h
    if "meta-llama" in model_name:
        return model.model.layers
    if "Qwen" in model_name:
        return model.model.layers
    if "kakaocorp" in model_name:
        return model.model.layers
    if "LGAI" in model_name:
        return model.transformer.h
    if "gemma-2" in model_name.lower():
        return model.base_model.layers
    if "gemma-3" in model_name.lower():
        return model.language_model.layers
    raise ValueError(f"Unsupported model: {model_name}")


def get_expected_num_layers(model_name: str, model_config: Any | None = None) -> int:
    if model_config is not None and hasattr(model_config, "num_hidden_layers"):
        return int(model_config.num_hidden_layers)

    known = {
        "meta-llama/Meta-Llama-3.1-8B-Instruct": 32,
        "meta-llama/Meta-Llama-3.1-8B": 32,
        "meta-llama/Llama-3.1-8B": 32,
        "Qwen/Qwen3-4B-Instruct-2507": 36,
        "google/gemma-3-4b-it": 34,
        "kakaocorp/kanana-1.5-8b-instruct-2505": 32,
        "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct": 30,
        "google/gemma-2-2b": 26,
    }
    if model_name in known:
        return known[model_name]
    raise ValueError(f"Unsupported model for layer count: {model_name}")


def get_mlp_up_proj(model_name: str, block: torch.nn.Module) -> torch.nn.Module:
    if "LGAI" in model_name:
        return block.mlp.c_fc_0
    return block.mlp.up_proj


def get_threshold_based_neuron_indices(correlation_scores: Any, at: int, threshold: float) -> list[torch.Tensor]:
    per_layer_indices: list[torch.Tensor] = []
    for layer_scores in correlation_scores[at]:
        per_layer_indices.append(torch.where(layer_scores > threshold)[0])
    return per_layer_indices


def load_correlation_scores(run4_base_dir: str, neuron_path: str, model_name: str) -> Any:
    pkl_path = os.path.join(run4_base_dir, neuron_path, model_name, "corr_results_all_at.pkl")
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"Neuron correlation file not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def register_steering_hooks(
    *,
    model: AutoModelForCausalLM,
    model_name: str,
    indices_pos: list[torch.Tensor],
    magnitude_alpha: float,
    indices_neg: list[torch.Tensor] | None = None,
) -> tuple[list[Any], list[int]]:
    blocks = get_llm_blocks(model, model_name)
    expected = get_expected_num_layers(model_name, getattr(model, "config", None))
    if len(blocks) != expected:
        raise RuntimeError(f"Unexpected layer count: found {len(blocks)} but expected {expected}")

    hook_handles: list[Any] = []
    num_selected_neurons: list[int] = []
    for layer_idx, block in enumerate(blocks):
        module = get_mlp_up_proj(model_name, block)
        device = next(module.parameters()).device

        selected = indices_pos[layer_idx]
        if indices_neg is not None:
            selected = selected[~torch.isin(selected, indices_neg[layer_idx])]

        hook = HiddenStateHook(
            neuron_indices=selected.to(device=device, dtype=torch.long),
            magnitude_alpha=magnitude_alpha,
        )
        handle = module.register_forward_hook(hook)
        hook_handles.append(handle)
        num_selected_neurons.append(int(selected.numel()))
    return hook_handles, num_selected_neurons


@torch.no_grad()
def generate_outputs(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
) -> list[dict[str, str]]:
    outputs: list[dict[str, str]] = []
    for start_idx in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[start_idx : start_idx + batch_size]
        batch = tokenizer(
            batch_prompts,
            padding=True,
            truncation=False,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(model.device)

        generated = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        prompt_lens = batch["attention_mask"].sum(dim=1)
        for row_idx in range(generated.shape[0]):
            prompt_len = int(prompt_lens[row_idx].item())
            outputs.append(
                {
                    "input_text": tokenizer.decode(batch["input_ids"][row_idx], skip_special_tokens=True),
                    "output_text": tokenizer.decode(generated[row_idx, prompt_len:], skip_special_tokens=True),
                }
            )
    return outputs


def save_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)
