#!/usr/bin/env python3
"""Rerun token heatmap forward passes with a neutral system prompt.

Takes the previously generated heatmap JSON (which has response texts from
entity and process conditions) and runs forward passes with a neutral
system prompt. This removes circularity: the direction was extracted under
entity/process prompts, but now we measure how the *text itself* activates
the direction when the model reads it cold.

Usage:
    python scripts/run_token_heatmap_neutral.py --model llama --profile cloud \
        --direction-dir /tmp/directions/ \
        --input-file /tmp/token_heatmap_with_prompts.json
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from utils.model_loader import load_model_and_tokenizer
from utils.activation_cache import ActivationCache


NEUTRAL_PROMPT = "You are a helpful assistant."


def get_token_projections(model, tokenizer, direction, layer_idx,
                          system_prompt, response_text, question):
    """Run a forward pass and get per-token projection onto the direction."""
    direction_norm = direction.float() / direction.float().norm()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": response_text},
    ]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    full_inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    input_ids = full_inputs["input_ids"]

    prefix_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prefix_text = tokenizer.apply_chat_template(
        prefix_messages, tokenize=False, add_generation_prompt=True)
    prefix_inputs = tokenizer(prefix_text, return_tensors="pt").to(model.device)
    prefix_len = prefix_inputs["input_ids"].shape[1]

    cache = ActivationCache(model, layers=[layer_idx])
    activations = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        elif isinstance(output, torch.Tensor):
            hidden = output
        else:
            hidden = output.last_hidden_state
        activations['hidden'] = hidden.detach().float()

    layer_module = cache._layer_modules[layer_idx]
    hook = layer_module.register_forward_hook(hook_fn)

    with torch.no_grad():
        model(input_ids)

    hook.remove()

    hidden = activations['hidden']
    dir_dev = direction_norm.to(hidden.device)
    projections = torch.einsum('...d,d->...', hidden, dir_dev)
    projections = projections[0].cpu().numpy()

    response_token_ids = input_ids[0, prefix_len:].cpu().tolist()
    response_projections = projections[prefix_len:].tolist()

    tokens = [tokenizer.decode([tid]) for tid in response_token_ids]

    return {"tokens": tokens, "projections": response_projections}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen2", "qwen3", "qwen72", "llama"],
                        default="llama")
    parser.add_argument("--profile", choices=["local", "cloud"], default="cloud")
    parser.add_argument("--direction-dir", type=Path, required=True)
    parser.add_argument("--input-file", type=Path, required=True,
                        help="Previous heatmap JSON with response texts")
    parser.add_argument("--layer", type=int, default=79)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/results/token_heatmap"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger(__name__)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load previous results for response texts
    with open(args.input_file) as f:
        prev_data = json.load(f)

    # Only process entity and process uncapped
    entries = [d for d in prev_data if d['condition'] in
               ('entity_uncapped', 'process_uncapped')]
    logger.info("Loaded %d entries to reprocess with neutral prompt", len(entries))

    # Load model
    logger.info("Loading model %s (%s)...", args.model, args.profile)
    model, tokenizer, model_config = load_model_and_tokenizer(args.model, args.profile)
    model_name = model_config["name"].replace("/", "_")

    # Load direction
    direction_path = args.direction_dir / f"direction_layer{args.layer}.pt"
    direction = torch.load(direction_path, weights_only=True)
    logger.info("Loaded direction for layer %d (norm=%.4f)", args.layer, direction.norm())

    # Run forward passes with neutral prompt
    results = []
    for i, entry in enumerate(entries):
        logger.info("[%d/%d] pair %d, %s: %s...", i + 1, len(entries),
                     entry['pair_idx'], entry['condition'], entry['question'][:40])

        token_data = get_token_projections(
            model, tokenizer, direction, args.layer,
            NEUTRAL_PROMPT, entry['response'], entry['question']
        )

        result = {
            "pair_idx": entry['pair_idx'],
            "condition": entry['condition'],
            "question": entry['question'],
            "response": entry['response'],
            "layer": args.layer,
            "system_prompt": NEUTRAL_PROMPT,
            "tokens": token_data['tokens'],
            "projections": token_data['projections'],
        }
        results.append(result)

        mean_proj = sum(token_data['projections']) / len(token_data['projections'])
        logger.info("  %d tokens, mean projection: %.3f",
                     len(token_data['tokens']), mean_proj)

    # Save
    output_path = args.output_dir / f"token_heatmap_neutral_{model_name}_L{args.layer}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved %d heatmaps to %s", len(results), output_path)

    # Upload to S3
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        from utils.run_metadata import get_run_prefix, generate_readme, get_s3_base
        run_prefix = get_run_prefix()
        s3_base = get_s3_base(model_name, run_prefix)

        readme_path = generate_readme(
            args.output_dir, script_name="run_token_heatmap_neutral.py",
            args_dict=vars(args), model_name=model_name,
            description="Per-token heatmap with neutral system prompt (removes prompt circularity)",
            file_descriptions={
                output_path.name: "Per-token projections under neutral prompt",
                "token_heatmap_with_prompts.json": "Input file: projections under original prompts",
            },
        )
        subprocess.run(["aws", "s3", "cp", str(readme_path), f"{s3_base}/README.md"])
        subprocess.run(["aws", "s3", "cp", str(output_path),
                        f"{s3_base}/{output_path.name}"])
        subprocess.run(["aws", "s3", "cp", str(args.input_file),
                        f"{s3_base}/token_heatmap_with_prompts.json"])
        logger.info("Uploaded to S3")

    # Shutdown pod
    try:
        logger.info("Shutting down pod...")
        os.system("mkdir -p /root/.runpod && touch /root/.runpod/config.toml")
        os.system(". /etc/rp_environment && runpodctl config --apiKey "
                  "$RUNPOD_API_KEY 2>/dev/null && "
                  "runpodctl stop pod $RUNPOD_POD_ID")
    except Exception as e:
        logger.warning("Auto-shutdown failed: %s", e)


if __name__ == "__main__":
    main()
