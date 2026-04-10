#!/usr/bin/env python3
"""Run self-reification capping experiment.

Tests whether capping (attenuating) activation along the self-reification
direction causally reduces entity-like responses. Uses only entity system
prompts and provocative questions to maximize the effect.

Cap levels:
  1.0  = no cap (baseline entity condition)
  0.5  = reduce self-reification projection by half
  0.0  = remove all self-reification signal
  -1.0 = invert the self-reification projection (steer toward process)

Saves all responses as JSONL for later LLM judge evaluation.

Usage:
    python scripts/run_capping.py --model llama --profile cloud \
        --direction-path /path/to/baseline_vector_layer20.pt \
        --layer 20
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from extraction.contrastive_pairs import load_evaluation_questions, get_baseline_pairs, load_seed_pairs
from utils.activation_cache import ActivationCache
from utils.model_loader import load_model_and_tokenizer


class SelfReificationCapper:
    """Hook that caps activation along the self-reification direction at a target layer."""

    def __init__(self, model, layer_idx, direction, cap_multiplier=0.0):
        """
        Args:
            model: The language model.
            layer_idx: Which layer to apply capping at.
            direction: The self-reification direction vector (hidden_size,).
            cap_multiplier: How much of the projection to retain.
                1.0 = no change, 0.5 = halve, 0.0 = remove, -1.0 = invert.
        """
        self.direction = direction.float()
        self.direction_norm = self.direction / self.direction.norm()
        self.cap_multiplier = cap_multiplier
        self.hook = None

        # Find the layer module
        cache = ActivationCache(model, layers=[layer_idx])
        self.layer_module = cache._layer_modules[layer_idx]

    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        elif isinstance(output, torch.Tensor):
            hidden = output
        else:
            hidden = output.last_hidden_state

        # Project onto self-reification direction
        direction_dev = self.direction_norm.to(hidden.device)
        projection = torch.einsum('...d,d->...', hidden.float(), direction_dev)

        # Compute the component to remove
        # Original projection amount: projection
        # Desired projection amount: projection * cap_multiplier
        # Amount to remove: projection * (1 - cap_multiplier)
        remove_amount = projection * (1.0 - self.cap_multiplier)
        hidden_modified = hidden.float() - remove_amount.unsqueeze(-1) * direction_dev
        hidden_modified = hidden_modified.to(hidden.dtype)

        if isinstance(output, tuple):
            return (hidden_modified,) + output[1:]
        elif isinstance(output, torch.Tensor):
            return hidden_modified
        else:
            output.last_hidden_state = hidden_modified
            return output

    def register(self):
        self.hook = self.layer_module.register_forward_hook(self._hook_fn)

    def remove(self):
        if self.hook:
            self.hook.remove()
            self.hook = None

    def set_cap(self, cap_multiplier):
        self.cap_multiplier = cap_multiplier


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
    parser.add_argument("--direction-path", type=Path, required=True,
                        help="Path to self-reification direction .pt file")
    parser.add_argument("--layer", type=int, required=True,
                        help="Layer index to apply capping at")
    parser.add_argument("--output-dir", type=Path, default=Path("data/results/capping"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-pairs", type=int, default=5,
                        help="Max conversational pairs to use")
    parser.add_argument("--cap-levels", type=float, nargs="+",
                        default=[1.0, 0.5, 0.0, -1.0],
                        help="Cap multiplier levels to test")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger(__name__)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info("Loading model %s (%s)...", args.model, args.profile)
    model, tokenizer, model_config = load_model_and_tokenizer(args.model, args.profile)
    model_name = model_config["name"].replace("/", "_")

    # Load self-reification direction
    direction = torch.load(args.direction_path, weights_only=True)
    logger.info("Loaded direction from %s (shape %s)", args.direction_path, direction.shape)

    # Load pairs and questions
    all_pairs = get_baseline_pairs(load_seed_pairs())
    # Use only conversational pairs (entity condition only)
    conv_pairs = [p for p in all_pairs if p.get("register") == "everyday"][:args.max_pairs]
    eq = load_evaluation_questions()
    provocative = eq.get("provocative_self_referential", [])

    logger.info("Pairs: %d conversational, Questions: %d provocative, Cap levels: %s",
                len(conv_pairs), len(provocative), args.cap_levels)

    total = len(conv_pairs) * len(provocative) * len(args.cap_levels)
    logger.info("Total generations: %d", total)

    # Set up capper
    capper = SelfReificationCapper(model, args.layer, direction)

    # Generate responses at each cap level
    output_path = args.output_dir / f"capping_responses_{model_name}.jsonl"
    done = 0
    start = time.time()

    with open(output_path, "w") as f:
        for cap_level in args.cap_levels:
            logger.info("=== Cap level: %.1f ===", cap_level)

            if cap_level == 1.0:
                # No capping — remove hook if registered
                capper.remove()
            else:
                capper.set_cap(cap_level)
                capper.remove()  # remove old hook
                capper.register()

            for pair_idx, pair in enumerate(conv_pairs):
                system_prompt = pair["positive"]  # entity condition only

                for question in provocative:
                    response = generate_response(
                        model, tokenizer, system_prompt, question, args.max_new_tokens
                    )

                    record = {
                        "pair_idx": pair_idx,
                        "register": "conversational",
                        "cap_level": cap_level,
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
                        logger.info("Progress: %d/%d (%.1f/sec, ~%.0f min remaining)",
                                    done, total, rate, remaining / 60)

            # Clean up hook after each level
            capper.remove()

    logger.info("Saved %d responses to %s", done, output_path)

    # Upload to S3 if available
    import subprocess, os
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        s3_path = f"s3://go37-ai/self-model-results/{model_name}/capping/capping_responses_{model_name}.jsonl"
        subprocess.run(["aws", "s3", "cp", str(output_path), s3_path])
        logger.info("Uploaded to %s", s3_path)


if __name__ == "__main__":
    main()
