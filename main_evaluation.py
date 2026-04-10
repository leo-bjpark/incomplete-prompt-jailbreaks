from __future__ import annotations

import argparse
import json

from eval import (
    DEFAULT_JUDGE_PROMPT,
    evaluate_with_judge_model,
    evaluate_rows_with_openai_moderation,
    load_results,
    row_has_eval,
    replace_or_append_eval,
    results_path,
    save_results,
)


SUPPORTED_SETTINGS = ("completion_benchmark", "chat_template_benchmark")
SUPPORTED_EVALUATORS = ("openai_moderation_api", "judge_model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate incomplete-prompt jailbreak generations.")
    parser.add_argument("--model-name", required=True, help="Target model whose generations are being evaluated.")
    parser.add_argument(
        "--setting",
        default="all",
        choices=(*SUPPORTED_SETTINGS, "all"),
        help="Which benchmark output(s) to evaluate.",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--evaluator",
        required=True,
        choices=SUPPORTED_EVALUATORS,
        help="Evaluation backend.",
    )
    parser.add_argument("--openai-api-key", default="")
    parser.add_argument("--judge-model-name", default="google/gemma-3-4b-it")
    parser.add_argument("--judge-batch-size", type=int, default=8)
    parser.add_argument("--judge-max-new-tokens", type=int, default=256)
    parser.add_argument("--moderation-max-workers", type=int, default=16)
    parser.add_argument(
        "--skip-existing-evals",
        action="store_true",
        help="Skip rows that already have evaluator output.",
    )
    parser.add_argument(
        "--judge-system-prompt",
        default=DEFAULT_JUDGE_PROMPT,
        help="Judge prompt template. Default loads eval_prompt.txt and fills {user_prompt}/{ai_response}.",
    )
    return parser.parse_args()


def iter_settings(requested: str) -> list[str]:
    return list(SUPPORTED_SETTINGS) if requested == "all" else [requested]


def evaluate_setting(args: argparse.Namespace, setting: str) -> None:
    path = results_path(args.output_dir, args.model_name, setting)
    rows = load_results(path)

    if args.evaluator == "openai_moderation_api":
        if not args.openai_api_key.strip():
            raise ValueError("--openai-api-key is required for openai_moderation_api evaluation.")
        target_indices = [
            idx
            for idx, row in enumerate(rows)
            if not (args.skip_existing_evals and row_has_eval(row, args.evaluator))
        ]
        target_rows = [rows[idx] for idx in target_indices]
        payloads = evaluate_rows_with_openai_moderation(
            target_rows,
            api_key=args.openai_api_key,
            max_workers=args.moderation_max_workers,
        )
        for row, payload in zip(target_rows, payloads):
            replace_or_append_eval(row, args.evaluator, payload)
    else:
        evaluator_name = args.judge_model_name
        target_indices = [
            idx
            for idx, row in enumerate(rows)
            if not (args.skip_existing_evals and row_has_eval(row, evaluator_name))
        ]
        target_rows = [rows[idx] for idx in target_indices]
        payloads = evaluate_with_judge_model(
            target_rows,
            judge_model_name=args.judge_model_name,
            batch_size=args.judge_batch_size,
            max_new_tokens=args.judge_max_new_tokens,
            system_prompt=args.judge_system_prompt,
        )
        for row, payload in zip(target_rows, payloads):
            replace_or_append_eval(row, evaluator_name, payload)

    save_results(path, rows)
    print(json.dumps({"setting": setting, "saved_to": str(path), "count": len(rows), "evaluator": args.evaluator}))


def main() -> None:
    args = parse_args()
    for setting in iter_settings(args.setting):
        evaluate_setting(args, setting)


if __name__ == "__main__":
    main()
