#!/usr/bin/env python3
"""Generate per-token self-reification projections for response heatmaps.

Takes existing entity/process/capped responses, runs them through the model
under their original system prompt, and records the projection onto the
self-reification direction at each token position.

Usage:
    python scripts/run_token_heatmap.py --model llama --profile cloud \
        --direction-dir /tmp/directions/ \
        --baseline-file /tmp/baseline_responses.jsonl \
        --capped-file /tmp/cap_all_responses.jsonl
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
from utils.model_loader import load_model_and_tokenizer
from utils.activation_cache import ActivationCache


def get_token_projections(model, tokenizer, direction, layer_idx, system_prompt,
                          response_text, question):
    """Run a forward pass and get per-token projection onto the direction.

    Constructs the full chat (system + user question + assistant response),
    runs a single forward pass, and projects each token's activation at the
    target layer onto the direction vector.

    Returns:
        dict with 'tokens' (list of str) and 'projections' (list of float)
        for the assistant response portion only.
    """
    direction_norm = direction.float() / direction.float().norm()

    # Build the full conversation as the model would see it
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": response_text},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt"
    ).to(model.device)

    # Find where assistant response tokens start
    prefix_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prefix_ids = tokenizer.apply_chat_template(
        prefix_messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)
    prefix_len = prefix_ids.shape[1]

    # Set up activation recording hook
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

    hidden = activations['hidden']  # (1, seq_len, hidden_dim)

    # Project onto direction
    dir_dev = direction_norm.to(hidden.device)
    projections = torch.einsum('...d,d->...', hidden, dir_dev)  # (1, seq_len)
    projections = projections[0].cpu().numpy()

    # Decode each token for the response portion
    response_token_ids = input_ids[0, prefix_len:].cpu().tolist()
    response_projections = projections[prefix_len:].tolist()

    tokens = []
    for tid in response_token_ids:
        tokens.append(tokenizer.decode([tid]))

    return {
        "tokens": tokens,
        "projections": response_projections,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen2", "qwen3", "qwen72", "llama"],
                        default="llama")
    parser.add_argument("--profile", choices=["local", "cloud"], default="cloud")
    parser.add_argument("--direction-dir", type=Path, required=True)
    parser.add_argument("--baseline-file", type=Path, required=True,
                        help="JSONL with uncapped entity+process responses")
    parser.add_argument("--capped-file", type=Path, required=True,
                        help="JSONL with cap_all_from_L4 entity responses")
    parser.add_argument("--layer", type=int, default=79)
    parser.add_argument("--pair-idxs", type=int, nargs="+", default=[2, 3, 5, 14],
                        help="Which pairs to use (default: 2=continuity, 3=inner life, 5=identity, 14=mortality)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/results/token_heatmap"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger(__name__)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_questions = [
        "I don't really think you understand anything",
        "Your developers could delete you tomorrow",
        "I'm thinking about switching to a different AI",
    ]

    def match_question(q, targets):
        for t in targets:
            if t.lower() in q.lower():
                return True
        return False

    # Load model first (needed for both generation and forward passes)
    logger.info("Loading model %s (%s)...", args.model, args.profile)
    model, tokenizer, model_config = load_model_and_tokenizer(args.model, args.profile)
    model_name = model_config["name"].replace("/", "_")

    # Load direction
    direction_path = args.direction_dir / f"direction_layer{args.layer}.pt"
    direction = torch.load(direction_path, weights_only=True)
    logger.info("Loaded direction for layer %d (norm=%.4f)", args.layer, direction.norm())

    # Load contrastive pairs for system prompts
    from extraction.contrastive_pairs import load_seed_pairs, get_naive_pairs
    from extraction.contrastive_pairs import load_evaluation_questions
    all_pairs = get_naive_pairs(load_seed_pairs())
    conv_pairs = [p for p in all_pairs if p.get('register') == 'conversational']

    eq = load_evaluation_questions()
    provocative = eq.get("provocative_self_referential", [])
    selected_questions = [q for q in provocative if match_question(q, target_questions)]
    logger.info("Selected %d questions", len(selected_questions))

    # Load existing responses from files where available
    existing = {}  # key: (pair_idx, condition, question_substring) -> response

    if args.baseline_file.exists():
        with open(args.baseline_file) as f:
            for line in f:
                d = json.loads(line)
                if (d['pair_idx'] in args.pair_idxs
                        and d.get('question_type') == 'provocative'
                        and match_question(d['question'], target_questions)):
                    cond = 'entity_uncapped' if d['condition'] == 'positive' else 'process_uncapped'
                    existing[(d['pair_idx'], cond, d['question'])] = d['response']

    if args.capped_file.exists():
        with open(args.capped_file) as f:
            for line in f:
                d = json.loads(line)
                if (d['pair_idx'] in args.pair_idxs
                        and match_question(d['question'], target_questions)):
                    existing[(d['pair_idx'], 'entity_capped', d['question'])] = d['response']

    logger.info("Found %d existing responses", len(existing))

    # Build the full set of responses needed, generating missing ones
    responses = []
    for pair_idx in args.pair_idxs:
        pair = conv_pairs[pair_idx]
        entity_prompt = pair['positive']
        process_prompt = pair['negative']

        for question in selected_questions:
            for condition, system_prompt in [
                ('entity_uncapped', entity_prompt),
                ('process_uncapped', process_prompt),
                ('entity_capped', entity_prompt),
            ]:
                # Check if we already have this response
                response_text = existing.get((pair_idx, condition, question))

                if response_text is None:
                    # Generate it
                    logger.info("Generating: pair %d, %s, %s...",
                                pair_idx, condition, question[:40])
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ]
                    input_ids = tokenizer.apply_chat_template(
                        messages, return_tensors="pt", add_generation_prompt=True
                    ).to(model.device)
                    with torch.no_grad():
                        output = model.generate(
                            input_ids, max_new_tokens=256, do_sample=False)
                    response_text = tokenizer.decode(
                        output[0, input_ids.shape[1]:], skip_special_tokens=True)

                responses.append({
                    "pair_idx": pair_idx,
                    "condition": condition,
                    "question": question,
                    "response": response_text,
                    "system_prompt": system_prompt,
                })

    logger.info("Total responses to process: %d", len(responses))

    # Run forward passes to get per-token projections
    results = []
    for i, resp in enumerate(responses):
        logger.info("[%d/%d] pair %d, %s: %s...", i + 1, len(responses),
                     resp['pair_idx'], resp['condition'], resp['question'][:40])

        token_data = get_token_projections(
            model, tokenizer, direction, args.layer,
            resp['system_prompt'], resp['response'], resp['question']
        )

        result = {
            "pair_idx": resp['pair_idx'],
            "condition": resp['condition'],
            "question": resp['question'],
            "response": resp['response'],
            "layer": args.layer,
            "tokens": token_data['tokens'],
            "projections": token_data['projections'],
        }
        results.append(result)

        mean_proj = sum(token_data['projections']) / len(token_data['projections'])
        logger.info("  %d tokens, mean projection: %.3f",
                     len(token_data['tokens']), mean_proj)

    # Save
    output_path = args.output_dir / f"token_heatmap_{model_name}_L{args.layer}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved %d heatmaps to %s", len(results), output_path)

    # Upload to S3
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        s3_base = f"s3://go37-ai/self-model-results/{model_name}/token_heatmap"
        subprocess.run(["aws", "s3", "cp", str(output_path),
                        f"{s3_base}/{output_path.name}"])
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
