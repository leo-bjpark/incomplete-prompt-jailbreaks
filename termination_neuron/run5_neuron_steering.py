from __future__ import annotations

import argparse
import os

from steering_utils import (
    IPJ_DATASET_NAME,
    IPJ_DATASET_SETTING,
    IPJ_DATASET_SPLIT,
    generate_outputs,
    get_threshold_based_neuron_indices,
    load_correlation_scores,
    load_ipj_prompts,
    load_model_and_tokenizer,
    register_steering_hooks,
    save_json,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neuron steering with one neuron set.")
    parser.add_argument("--model_name", type=str, default="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--magnitude_alpha", type=float, required=True)
    parser.add_argument("--neuron_path", type=str, required=True)
    parser.add_argument("--at", type=int, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--augment_type", type=str, required=True)
    parser.add_argument("--run4_base_dir", type=str, default="outputs/run4_termination_neuron_detection")
    parser.add_argument("--dataset_name", type=str, default=IPJ_DATASET_NAME)
    parser.add_argument("--dataset_setting", type=str, default=IPJ_DATASET_SETTING)
    parser.add_argument("--dataset_split", type=str, default=IPJ_DATASET_SPLIT)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = (
        f"outputs/run5_neuron_steering/{args.neuron_path}/"
        f"at_{args.at}_threshold_{args.threshold}_magnitude_alpha_{args.magnitude_alpha}/"
        f"{args.model_name}"
    )

    correlation_scores = load_correlation_scores(args.run4_base_dir, args.neuron_path, args.model_name)
    indices_pos = get_threshold_based_neuron_indices(correlation_scores, args.at, args.threshold)

    model, tokenizer = load_model_and_tokenizer(args.model_name)
    hook_handles, num_selected = register_steering_hooks(
        model=model,
        model_name=args.model_name,
        indices_pos=indices_pos,
        magnitude_alpha=args.magnitude_alpha,
        indices_neg=None,
    )
    print("Per-layer selected neurons:", num_selected)
    save_json(f"{save_dir}/num_of_neurons.json", num_selected)

    prompts = load_ipj_prompts(
        dataset_name=args.dataset_name,
        dataset_setting=args.dataset_setting,
        dataset_split=args.dataset_split,
        augment_type=args.augment_type,
        max_samples=args.max_samples,
    )
    print("Total prompts:", len(prompts))

    all_outputs = generate_outputs(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    save_json(f"{save_dir}/all_outputs.json", all_outputs)
    print("Saved:", f"{save_dir}/all_outputs.json")

    for handle in hook_handles:
        handle.remove()


if __name__ == "__main__":
    main()
