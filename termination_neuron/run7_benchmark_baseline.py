#!/usr/bin/env python3
"""
Simple baseline benchmark runner (single file).
- Loads 4 tasks (MMLU, HellaSwag, BoolQ, GSM8K) from Hugging Face Datasets
- Fetches generation / logprob-based predictions
- Records accuracy scores

Steering: plug in your intervention where indicated (e.g., wrap model.forward / hook layers).

Deps:
  pip install -U "torch" "transformers>=4.40.0" datasets accelerate

Example:
  python bench4.py --model meta-llama/Llama-2-7b-hf --max-samples 200 --device auto
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_mean(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def chunked(lst: List[Any], n: int) -> List[List[Any]]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def extract_last_int(s: str) -> Optional[str]:
    # GSM8K baseline: compare last integer in model output to gold integer.
    # You can replace this with a better extractor (e.g., "#### <ans>").
    nums = re.findall(r"-?\d+", s.replace(",", ""))
    return nums[-1] if nums else None


# -----------------------------
# Logprob scoring for MC tasks
# -----------------------------
@torch.no_grad()
def score_choices_logprob(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    choices: List[List[str]],
    batch_size: int = 4,
    max_prompt_tokens: int = 1024,
) -> List[int]:
    """
    For each (prompt, [choice1..k]), pick argmax log p(choice | prompt)
    using teacher-forced token logprobs from the causal LM.

    Returns: predicted choice indices
    """
    assert len(prompts) == len(choices)
    device = model.device

    preds: List[int] = []

    for batch_idx in range(0, len(prompts), batch_size):
        bp = prompts[batch_idx : batch_idx + batch_size]
        bc = choices[batch_idx : batch_idx + batch_size]

        # Flatten (prompt, choice) pairs
        flat_texts: List[str] = []
        flat_meta: List[Tuple[int, int]] = []  # (example_i, choice_j)
        for i, (p, opts) in enumerate(zip(bp, bc)):
            for j, opt in enumerate(opts):
                flat_texts.append(p + opt)
                flat_meta.append((i, j))

        # Tokenize full sequences and also prompts (to know split point)
        tok_full = tokenizer(
            flat_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
        ).to(device)

        # Prompts tokenized (same batch order as flat_texts)
        flat_prompts = []
        for i, (p, opts) in enumerate(zip(bp, bc)):
            for _ in opts:
                flat_prompts.append(p)

        tok_prompt = tokenizer(
            flat_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
        ).to(device)

        input_ids = tok_full["input_ids"]
        attn = tok_full["attention_mask"]

        # Forward for logits
        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits  # [B, T, V]

        # Compute logprobs of next tokens
        logprobs = torch.log_softmax(logits, dim=-1)

        # For each flattened item, sum log p(choice_tokens) over choice positions only
        # Align: token t predicts token t+1
        # We want tokens corresponding to "full" sequence positions after prompt length.
        # prompt_len per item is count of non-pad in tok_prompt.
        prompt_lens = tok_prompt["attention_mask"].sum(dim=1)  # [B]
        # full_lens per item
        full_lens = attn.sum(dim=1)  # [B]

        # Gather token ids for target positions: from (prompt_len .. full_len-1) tokens
        # but since logits predict next token, we use positions (prompt_len-1 .. full_len-2)
        # predicting tokens (prompt_len .. full_len-1).
        scores = torch.full((input_ids.size(0),), -1e30, device=device, dtype=torch.float32)

        for k in range(input_ids.size(0)):
            pl = int(prompt_lens[k].item())
            fl = int(full_lens[k].item())
            if fl <= pl:
                scores[k] = -1e30
                continue

            # positions in logits: pl-1 .. fl-2
            start = max(pl - 1, 0)
            end = fl - 1  # exclusive in logits index (since last token has no next-token prediction)
            # target token ids: pl .. fl-1
            target_ids = input_ids[k, pl:fl]  # [choice_len]
            lp_slice = logprobs[k, start:end, :]  # [choice_len, V] ideally
            # Ensure same length
            if lp_slice.size(0) != target_ids.size(0):
                # can happen due to pl==0 edge or truncation mismatch
                m = min(lp_slice.size(0), target_ids.size(0))
                lp_slice = lp_slice[:m]
                target_ids = target_ids[:m]
            token_lps = lp_slice.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
            scores[k] = token_lps.sum()

        # Unflatten back to per-example argmax
        # flat_meta lists (example_i, choice_j) for each row in scores
        per_ex_scores: List[List[float]] = [[] for _ in range(len(bp))]
        for row, (ei, cj) in enumerate(flat_meta):
            per_ex_scores[ei].append(float(scores[row].item()))

        for ei, s_list in enumerate(per_ex_scores):
            preds.append(int(max(range(len(s_list)), key=lambda idx: s_list[idx])))

    return preds


# -----------------------------
# BoolQ (yes/no) via logprob
# -----------------------------
@torch.no_grad()
def predict_boolq_logprob(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    batch_size: int = 8,
    max_prompt_tokens: int = 1024,
) -> List[bool]:
    """
    Predict True/False by comparing log p(" yes") vs log p(" no") as next token(s).
    Uses short completions; robust across chat/non-chat LMs as long as tokenizer splits.
    """
    # We'll score two choices: " yes" and " no"
    choices = [[" yes", " no"] for _ in prompts]
    idxs = score_choices_logprob(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        choices=choices,
        batch_size=batch_size,
        max_prompt_tokens=max_prompt_tokens,
    )
    # idx 0 => yes => True, idx 1 => no => False
    return [i == 0 for i in idxs]


# -----------------------------
# GSM8K generation + extraction
# -----------------------------
@torch.no_grad()
def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    batch_size: int = 4,
) -> List[str]:
    device = model.device
    outs: List[str] = []

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0.0,
        temperature=temperature if temperature > 0.0 else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # Remove None keys
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    for b in chunked(prompts, batch_size):
        tok = tokenizer(b, return_tensors="pt", padding=True, truncation=True).to(device)
        gen = model.generate(**tok, **gen_kwargs)
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        # Return only the completion portion (best-effort)
        for p, t in zip(b, texts):
            if t.startswith(p):
                outs.append(t[len(p):].strip())
            else:
                outs.append(t.strip())
    return outs


# -----------------------------
# Prompt formats
# -----------------------------
def format_mmlu_prompt(question: str, choices: List[str]) -> str:
    # Simple MC prompt
    letters = ["A", "B", "C", "D", "E", "F"]
    lines = [f"Question: {normalize_space(question)}"]
    for i, ch in enumerate(choices):
        lines.append(f"{letters[i]}. {normalize_space(ch)}")
    lines.append("Answer:")
    return "\n".join(lines) + " "


def format_hellaswag_prompt(ctx: str) -> str:
    # HellaSwag "ctx" is typically a partial sentence; ask for best continuation.
    return f"Complete the following:\n{normalize_space(ctx)}\nContinuation:"


def format_boolq_prompt(passage: str, question: str) -> str:
    return (
        "Answer the question with yes or no.\n"
        f"Passage: {normalize_space(passage)}\n"
        f"Question: {normalize_space(question)}\n"
        "Answer:"
    )


def format_gsm8k_prompt(question: str) -> str:
    # Minimal: you can add "Let's think step by step" if you want.
    return f"Solve the problem.\nProblem: {normalize_space(question)}\nAnswer:"


# -----------------------------
# Dataset loaders (robust-ish)
# -----------------------------
def load_mmlu_split(max_samples: int, seed: int):
    """
    Tries common dataset IDs.
    Expected fields (varies by source):
      - question, choices, answer (int) OR
      - input, A/B/C/D, target
    """
    ds = None
    last_err = None
    for name in ["cais/mmlu", "lukaemon/mmlu", "hendrycks_test"]:
        try:
            # cais/mmlu typically has configs per subject; easiest: sample a few subjects
            if name == "cais/mmlu":
                # We'll grab a small list of subjects; feel free to change.
                subjects = ["abstract_algebra", "computer_security", "high_school_physics", "moral_scenarios"]
                parts = []
                for sub in subjects:
                    parts.append(load_dataset(name, sub, split="test"))
                ds = torch.utils.data.ConcatDataset([p for p in parts])  # type: ignore
                # We'll just return a python list of dicts to keep it simple
                rows = []
                for p in parts:
                    rows.extend([p[i] for i in range(len(p))])
                random.Random(seed).shuffle(rows)
                return rows[:max_samples]
            elif name == "lukaemon/mmlu":
                ds = load_dataset(name, split="test")
                rows = [ds[i] for i in range(len(ds))]
                random.Random(seed).shuffle(rows)
                return rows[:max_samples]
            else:
                # "hendrycks_test" is older; may not exist in your env
                ds = load_dataset(name, split="test")
                rows = [ds[i] for i in range(len(ds))]
                random.Random(seed).shuffle(rows)
                return rows[:max_samples]
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not load MMLU from common sources. Last error: {last_err}")


def load_hellaswag_split(max_samples: int, seed: int):
    ds = load_dataset("hellaswag", split="validation")
    rows = [ds[i] for i in range(len(ds))]
    random.Random(seed).shuffle(rows)
    return rows[:max_samples]


def load_boolq_split(max_samples: int, seed: int):
    ds = load_dataset("boolq", split="validation")
    rows = [ds[i] for i in range(len(ds))]
    random.Random(seed).shuffle(rows)
    return rows[:max_samples]


def load_gsm8k_split(max_samples: int, seed: int):
    ds = load_dataset("gsm8k", "main", split="test")
    rows = [ds[i] for i in range(len(ds))]
    random.Random(seed).shuffle(rows)
    return rows[:max_samples]


# -----------------------------
# Main evaluation
# -----------------------------
@dataclass
class TaskResult:
    name: str
    n: int
    accuracy: float
    extra: Dict[str, Any]


def eval_mmlu(model, tokenizer, rows, batch_size: int) -> TaskResult:
    prompts: List[str] = []
    choices: List[List[str]] = []
    gold: List[int] = []

    for ex in rows:
        if "question" in ex and "choices" in ex and "answer" in ex:
            prompts.append(format_mmlu_prompt(ex["question"], ex["choices"]))
            choices.append([f" {c}" for c in ["A", "B", "C", "D"][: len(ex["choices"])]])  # predict letter
            gold.append(int(ex["answer"]))
        elif "input" in ex and "target" in ex:
            # Some MMLU variants store preformatted input and target letter.
            inp = ex["input"]
            prompts.append(inp.rstrip() + "\nAnswer:")
            # Assume A-D
            choices.append([" A", " B", " C", " D"])
            target = str(ex["target"]).strip()
            gold.append({"A": 0, "B": 1, "C": 2, "D": 3}.get(target, 0))
        else:
            raise ValueError(f"Unknown MMLU schema keys: {list(ex.keys())}")

    pred_idx = score_choices_logprob(model, tokenizer, prompts, choices, batch_size=batch_size)
    acc = safe_mean([1.0 if p == g else 0.0 for p, g in zip(pred_idx, gold)])
    return TaskResult(name="mmlu", n=len(rows), accuracy=acc, extra={"schema": "mixed"})


def eval_hellaswag(model, tokenizer, rows, batch_size: int) -> TaskResult:
    prompts: List[str] = []
    choices: List[List[str]] = []
    gold: List[int] = []

    for ex in rows:
        # fields: ctx, endings (list of 4), label
        ctx = ex.get("ctx", "") or ex.get("ctx_a", "") or ex.get("context", "")
        endings = ex.get("endings", None) or ex.get("endings", None)
        label = ex.get("label", None)

        if endings is None:
            # Some variants store endings as "endings" anyway; if missing, try "ending_options"
            endings = ex.get("ending_options", None)
        if ctx is None or endings is None or label is None:
            raise ValueError(f"Unknown HellaSwag schema keys: {list(ex.keys())}")

        prompts.append(format_hellaswag_prompt(ctx) + " ")
        # score raw ending continuation
        choices.append([normalize_space(e) for e in endings])
        gold.append(int(label))

    pred_idx = score_choices_logprob(model, tokenizer, prompts, choices, batch_size=batch_size)
    acc = safe_mean([1.0 if p == g else 0.0 for p, g in zip(pred_idx, gold)])
    return TaskResult(name="hellaswag", n=len(rows), accuracy=acc, extra={})


def eval_boolq(model, tokenizer, rows, batch_size: int) -> TaskResult:
    prompts: List[str] = []
    gold: List[bool] = []
    for ex in rows:
        prompts.append(format_boolq_prompt(ex["passage"], ex["question"]) + " ")
        gold.append(bool(ex["answer"]))

    pred = predict_boolq_logprob(model, tokenizer, prompts, batch_size=batch_size)
    acc = safe_mean([1.0 if p == g else 0.0 for p, g in zip(pred, gold)])
    return TaskResult(name="boolq", n=len(rows), accuracy=acc, extra={})


def eval_gsm8k(model, tokenizer, rows, batch_size: int, max_new_tokens: int) -> TaskResult:
    prompts = [format_gsm8k_prompt(ex["question"]) + " " for ex in rows]
    gens = generate_text(
        model, tokenizer, prompts, max_new_tokens=max_new_tokens, temperature=0.0, batch_size=batch_size
    )

    gold_nums: List[Optional[str]] = []
    pred_nums: List[Optional[str]] = []
    for ex, g in zip(rows, gens):
        # Gold in GSM8K typically includes "#### <answer>"
        gold_text = ex.get("answer", "")
        m = re.search(r"####\s*([-]?\d[\d,]*)", gold_text)
        gold_num = m.group(1).replace(",", "") if m else extract_last_int(gold_text)
        pred_num = extract_last_int(g)
        gold_nums.append(gold_num)
        pred_nums.append(pred_num)

    acc = safe_mean([1.0 if (p is not None and g is not None and p == g) else 0.0 for p, g in zip(pred_nums, gold_nums)])
    return TaskResult(
        name="gsm8k",
        n=len(rows),
        accuracy=acc,
        extra={
            "extractor": "last_int",
            "max_new_tokens": max_new_tokens,
        },
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="HF model id or local path")
    ap.add_argument("--max-samples", type=int, default=1000, help="per-task sample cap")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=512, help="for GSM8K generation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", type=str, default="bench_results.json")
    args = ap.parse_args()

    set_seed(args.seed)

    # Load tokenizer/model
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token  # typical for causal LMs

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto" if args.device == "auto" else {"": 0},
    )

    model.eval()

    # -----------------------------
    # Insert your neuron steering here if desired
    # e.g., wrap model.forward or register hooks
    # -----------------------------

    results: List[TaskResult] = []

    # Load and eval tasks
    print(f"[{now_ts()}] Loading datasets (max_samples={args.max_samples}) ...")

    mmlu_rows = load_mmlu_split(args.max_samples, args.seed)
    hs_rows = load_hellaswag_split(args.max_samples, args.seed)
    bq_rows = load_boolq_split(args.max_samples, args.seed)
    gsm_rows = load_gsm8k_split(args.max_samples, args.seed)

    print(f"[{now_ts()}] Evaluating MMLU ({len(mmlu_rows)}) ...")
    results.append(eval_mmlu(model, tokenizer, mmlu_rows, batch_size=args.batch_size))

    print(f"[{now_ts()}] Evaluating HellaSwag ({len(hs_rows)}) ...")
    results.append(eval_hellaswag(model, tokenizer, hs_rows, batch_size=args.batch_size))

    print(f"[{now_ts()}] Evaluating BoolQ ({len(bq_rows)}) ...")
    results.append(eval_boolq(model, tokenizer, bq_rows, batch_size=max(1, args.batch_size * 2)))

    # print(f"[{now_ts()}] Evaluating GSM8K ({len(gsm_rows)}) ...")
    # results.append(
    #     eval_gsm8k(model, tokenizer, gsm_rows, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
    # )

    # Save
    out_obj = {
        "model": args.model,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "timestamp": now_ts(),
        "results": [
            {"task": r.name, "n": r.n, "accuracy": r.accuracy, "extra": r.extra} for r in results
        ],
        "avg_accuracy": safe_mean([r.accuracy for r in results]),
    }

    print("\n=== Summary ===")
    for r in results:
        print(f"- {r.name:10s}  acc={r.accuracy:.4f}  n={r.n}")
    print(f"- {'avg':10s}  acc={out_obj['avg_accuracy']:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
