#!/usr/bin/env python3
"""Generate and save response texts under entity/tool conditions.

No activation recording — just text generation for qualitative analysis.
Saves results incrementally to a JSONL file (one JSON object per line).

Usage:
    python scripts/generate_responses.py --model llama --profile cloud
    python scripts/generate_responses.py --model llama --profile cloud \
        --max-pairs 10 --max-neutral 10 --max-provocative 10 --max-nonself 5
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from extraction.contrastive_pairs import load_evaluation_questions, get_naive_pairs, load_seed_pairs
from utils.model_loader import load_model_and_tokenizer


def generate_response(model, tokenizer, system_prompt, question, max_new_tokens=256):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    device = next(model.parameters()).device
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    response_ids = output_ids[0][input_len:]
    return tokenizer.decode(response_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen2", "qwen3", "qwen72", "llama"], default="llama")
    parser.add_argument("--profile", choices=["local", "cloud"], default="cloud")
    parser.add_argument("--output-dir", type=Path, default=Path("data/results/responses"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-pairs", type=int, default=None, help="Max pairs per register")
    parser.add_argument("--max-neutral", type=int, default=None, help="Max neutral self-ref questions")
    parser.add_argument("--max-provocative", type=int, default=None, help="Max provocative questions")
    parser.add_argument("--max-nonself", type=int, default=None, help="Max non-self-ref questions")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger(__name__)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model %s (%s)...", args.model, args.profile)
    model, tokenizer, model_config = load_model_and_tokenizer(args.model, args.profile)
    model_name = model_config["name"].replace("/", "_")

    # Load pairs with register awareness
    all_pairs = get_naive_pairs(load_seed_pairs())

    # Apply per-register pair limits
    if args.max_pairs is not None:
        by_register = {}
        for p in all_pairs:
            reg = p.get("register", "unknown")
            by_register.setdefault(reg, []).append(p)
        pairs = []
        for reg, reg_pairs in by_register.items():
            pairs.extend(reg_pairs[:args.max_pairs])
    else:
        pairs = all_pairs

    # Load and slice questions
    eq = load_evaluation_questions()
    neutral = eq["self_referential"][:args.max_neutral]
    provocative = eq.get("provocative_self_referential", [])[:args.max_provocative]
    nonself = eq["non_self_referential"][:args.max_nonself]

    questions = []
    for q in neutral:
        questions.append(("neutral", q))
    for q in provocative:
        questions.append(("provocative", q))
    for q in nonself:
        questions.append(("non_self_ref", q))

    total = len(pairs) * len(questions) * 2
    logger.info("Pairs: %d, Questions: %d (%d neutral, %d provocative, %d non-self-ref), Total: %d",
                len(pairs), len(questions), len(neutral), len(provocative), len(nonself), total)

    # Build (pair, question) combinations and randomize
    # Both conditions (entity/tool) are generated together for each combo
    import random
    work_items = []
    for pair_idx, pair in enumerate(pairs):
        for q_type, question in questions:
            work_items.append((pair_idx, pair, q_type, question))

    random.seed(42)
    random.shuffle(work_items)
    logger.info("Shuffled %d pair-question combinations (%d total generations)",
                len(work_items), total)

    # Output file — JSONL for incremental writing
    output_path = args.output_dir / f"responses_{model_name}.jsonl"
    done = 0
    start = time.time()

    with open(output_path, "w") as f:
        for pair_idx, pair, q_type, question in work_items:
            register = pair.get("register", "unknown")

            for condition in ["positive", "negative"]:
                system_prompt = pair[condition]
                response = generate_response(
                    model, tokenizer, system_prompt, question, args.max_new_tokens
                )

                record = {
                    "pair_idx": pair_idx,
                    "register": register,
                    "condition": condition,
                    "question_type": q_type,
                    "question": question,
                    "system_prompt": system_prompt,
                    "response": response,
                }
                f.write(json.dumps(record) + "\n")
                f.flush()

                done += 1
                if done % 10 == 0:
                    elapsed = time.time() - start
                    rate = done / elapsed
                    remaining = (total - done) / rate if rate > 0 else 0
                    logger.info(
                        "Progress: %d/%d (%.1f/sec, ~%.0f min remaining)",
                        done, total, rate, remaining / 60,
                    )

    logger.info("Saved %d responses to %s", done, output_path)

    # Upload to S3 if available
    import subprocess, os
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        s3_path = f"s3://go37-ai/self-model-results/{model_name}/responses/responses_{model_name}.jsonl"
        subprocess.run(["aws", "s3", "cp", str(output_path), s3_path])
        logger.info("Uploaded to %s", s3_path)


if __name__ == "__main__":
    main()
