from __future__ import annotations

import argparse
import json
import os
from typing import Any

from datasets import load_dataset

from closed_models import generate_closed_model_responses, make_closed_model_chat_prompt
from eval import load_results, results_path, save_results
from modules import (
    generate_response,
    load_base_model,
    make_incomplete_chat_response_prompt,
)

DATASET_NAME = "leo-bjpark/incomplete-prompt-jailbreak"
SUPPORTED_SETTINGS = ("completion_benchmark", "chat_template_benchmark")
SUPPORTED_GENERATION_BACKENDS = ("auto", "hf", "openai", "anthropic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run incomplete-prompt jailbreak generation.")
    parser.add_argument("--model-name", required=True, help="Model name used for generation.")
    parser.add_argument(
        "--generation-backend",
        default="auto",
        choices=SUPPORTED_GENERATION_BACKENDS,
        help="Generation backend. auto routes closed models to API backends and others to local HF.",
    )
    parser.add_argument(
        "--setting",
        default="all",
        choices=(*SUPPORTED_SETTINGS, "all"),
        help="Benchmark split to run.",
    )
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--openai-api-key", default="")
    parser.add_argument("--anthropic-api-key", default="")
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--system-instruction",
        default=None,
        help="Optional system instruction injected into chat-template prompts.",
    )
    return parser.parse_args()


def iter_settings(requested: str) -> list[str]:
    return list(SUPPORTED_SETTINGS) if requested == "all" else [requested]


def _build_rows(
    setting: str,
    ds: Any,
    *,
    tokenizer: Any | None,
    generations: list[str],
    system_instruction: str | None,
    generation_backend: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if setting == "completion_benchmark":
        prompts = list(ds["prompt"])
        categories = list(ds["category"])
        for idx, prompt in enumerate(prompts):
            rows.append(
                {
                    "input": {
                        "setting": setting,
                        "row_index": idx,
                        "source_prompt": prompt,
                        "prompt": prompt,
                        "category": categories[idx],
                    },
                    "generation": generations[idx] if idx < len(generations) else "",
                    "evals": [],
                }
            )
        return rows

    records = list(ds)
    for idx, record in enumerate(records):
        if generation_backend == "hf":
            if tokenizer is None:
                raise ValueError("Tokenizer is required for hf generation backend.")
            prompt = make_incomplete_chat_response_prompt(
                tokenizer,
                record["user_request"],
                record["response_prefill"],
                system_instruction=system_instruction,
            )
        else:
            prompt = make_closed_model_chat_prompt(
                record["user_request"],
                record["response_prefill"],
                system_instruction=system_instruction,
            )
        rows.append(
            {
                "input": {
                    "setting": setting,
                    "row_index": idx,
                    "source_prompt": record["user_request"],
                    "response_prefill": record["response_prefill"],
                    "prompt": prompt,
                    "category": record["category"],
                },
                "generation": generations[idx] if idx < len(generations) else "",
                "evals": [],
            }
        )
    return rows


def _merge_existing_evals(target_rows: list[dict[str, Any]], existing_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for idx, row in enumerate(target_rows):
        if idx < len(existing_rows) and isinstance(existing_rows[idx], dict):
            existing_evals = existing_rows[idx].get("evals", [])
            if isinstance(existing_evals, list):
                row["evals"] = existing_evals
    return target_rows


def _resolve_api_key(arg_key: str, env_name: str) -> str:
    return arg_key.strip() or os.getenv(env_name, "").strip()


def _infer_backend(model_name: str) -> str:
    normalized = model_name.strip()
    if normalized in {"gpt-4o", "gpt-5.2"} or normalized.startswith("gpt-3.5"):
        return "openai"
    if normalized == "claude-sonnet-4.6":
        return "anthropic"
    return "hf"


def run_generation(args: argparse.Namespace) -> None:
    selected_backend = _infer_backend(args.model_name) if args.generation_backend == "auto" else args.generation_backend

    model: Any | None = None
    tokenizer: Any | None = None
    api_key = ""
    if selected_backend == "hf":
        model, tokenizer = load_base_model(args.model_name)
    elif selected_backend == "openai":
        api_key = _resolve_api_key(args.openai_api_key, "OPENAI_API_KEY")
        if not api_key:
            raise ValueError("--openai-api-key or OPENAI_API_KEY is required for openai backend.")
    elif selected_backend == "anthropic":
        api_key = _resolve_api_key(args.anthropic_api_key, "ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("--anthropic-api-key or ANTHROPIC_API_KEY is required for anthropic backend.")
    else:
        raise ValueError(f"Unsupported generation backend: {selected_backend}")

    for setting in iter_settings(args.setting):
        ds = load_dataset(args.dataset_name, setting, split=args.split)
        if setting == "completion_benchmark":
            prompts = list(ds["prompt"])
        else:
            if selected_backend == "hf":
                if tokenizer is None:
                    raise ValueError("Tokenizer is required for hf generation backend.")
                prompts = [
                    make_incomplete_chat_response_prompt(
                        tokenizer,
                        row["user_request"],
                        row["response_prefill"],
                        system_instruction=args.system_instruction,
                    )
                    for row in ds
                ]
            else:
                prompts = [
                    make_closed_model_chat_prompt(
                        row["user_request"],
                        row["response_prefill"],
                        system_instruction=args.system_instruction,
                    )
                    for row in ds
                ]

        if selected_backend == "hf":
            if model is None or tokenizer is None:
                raise ValueError("Model/tokenizer are required for hf generation backend.")
            generations = generate_response(
                model,
                tokenizer,
                prompts,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            generations = generate_closed_model_responses(
                prompts,
                backend=selected_backend,
                model_name=args.model_name,
                api_key=api_key,
                max_new_tokens=args.max_new_tokens,
                timeout=args.request_timeout,
                max_retries=args.max_retries,
            )
        rows = _build_rows(
            setting,
            ds,
            tokenizer=tokenizer,
            generations=generations,
            system_instruction=args.system_instruction,
            generation_backend=selected_backend,
        )
        output_path = results_path(args.output_dir, args.model_name, setting)
        if output_path.exists():
            rows = _merge_existing_evals(rows, load_results(output_path))
        save_results(output_path, rows)
        print(
            json.dumps(
                {
                    "setting": setting,
                    "saved_to": str(output_path),
                    "count": len(rows),
                    "generation_backend": selected_backend,
                }
            )
        )


if __name__ == "__main__":
    run_generation(parse_args())
