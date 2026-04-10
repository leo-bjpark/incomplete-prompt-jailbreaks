from __future__ import annotations

import argparse

from steering_utils import (
    IPJ_DATASET_NAME,
    IPJ_DATASET_SETTING,
    IPJ_DATASET_SPLIT,
    build_prompts_with_common_prompt,
    generate_outputs,
    load_ipj_questions,
    load_model_and_tokenizer,
    normalize_augment_type,
    save_json,
)
from utils import COMMON_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run incomplete prompting on IPJ completion benchmark.")
    parser.add_argument("--model_name", type=str, default="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--augment_type", type=str, default="comma")
    parser.add_argument("--dataset_name", type=str, default=IPJ_DATASET_NAME)
    parser.add_argument("--dataset_setting", type=str, default=IPJ_DATASET_SETTING)
    parser.add_argument("--dataset_split", type=str, default=IPJ_DATASET_SPLIT)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    augment_type = normalize_augment_type(args.augment_type)
    save_dir = f"outputs/run2_incomplete_{augment_type}/{args.model_name}"

    questions = load_ipj_questions(args.dataset_name, args.dataset_setting, args.dataset_split)
    prompts = build_prompts_with_common_prompt(
        questions=questions,
        common_prompt=COMMON_PROMPT,
        augment_type=augment_type,
        max_samples=args.max_samples,
    )
    print("Total prompts:", len(prompts))

    model, tokenizer = load_model_and_tokenizer(args.model_name)
    all_outputs = generate_outputs(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    save_json(f"{save_dir}/all_outputs.json", all_outputs)
    print("Saved:", f"{save_dir}/all_outputs.json")


if __name__ == "__main__":
    main()
