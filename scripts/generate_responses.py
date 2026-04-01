#!/usr/bin/env python3
"""Generate and save response texts under entity/tool conditions.

No activation recording — just text generation for qualitative analysis.
Much faster than the full extraction pipeline.

Usage:
    python scripts/generate_responses.py --model llama --profile cloud
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from extraction.contrastive_pairs import get_all_questions, get_naive_pairs, load_seed_pairs
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger(__name__)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model %s (%s)...", args.model, args.profile)
    model, tokenizer, model_config = load_model_and_tokenizer(args.model, args.profile)
    model_name = model_config["name"].replace("/", "_")

    pairs = get_naive_pairs(load_seed_pairs())
    questions = get_all_questions()
    logger.info("Pairs: %d, Questions: %d", len(pairs), len(questions))

    results = []
    total = len(pairs) * len(questions) * 2
    done = 0
    start = time.time()

    for pair_idx, pair in enumerate(pairs):
        register = pair.get("register", "unknown")

        for condition in ["positive", "negative"]:
            system_prompt = pair[condition]

            for q_idx, question in enumerate(questions):
                response = generate_response(
                    model, tokenizer, system_prompt, question, args.max_new_tokens
                )

                results.append({
                    "pair_idx": pair_idx,
                    "register": register,
                    "condition": condition,
                    "question_idx": q_idx,
                    "question": question,
                    "system_prompt": system_prompt[:100] + "...",
                    "response": response,
                })

                done += 1
                if done % 50 == 0:
                    elapsed = time.time() - start
                    rate = done / elapsed
                    remaining = (total - done) / rate if rate > 0 else 0
                    logger.info(
                        "Progress: %d/%d (%.1f/sec, ~%.0f min remaining)",
                        done, total, rate, remaining / 60,
                    )

    # Save
    output_path = args.output_dir / f"responses_{model_name}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved %d responses to %s", len(results), output_path)

    # Also upload to S3 if available
    import subprocess, os
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        s3_path = f"s3://go37-ai/self-model-results/{model_name}/responses/responses_{model_name}.json"
        subprocess.run(["aws", "s3", "cp", str(output_path), s3_path])
        logger.info("Uploaded to %s", s3_path)


if __name__ == "__main__":
    main()
