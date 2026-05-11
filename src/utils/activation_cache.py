"""Record and store model activations at specified layers.

Provides an ActivationCache class that registers forward hooks on
transformer layers to capture hidden state activations during inference.
Used by all experiments to record the internal representations that are
then projected onto feature directions.

For MoE models, a sibling RouterCache captures per-token softmax
distributions over experts at each layer.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class ActivationCache:
    """Captures hidden-state activations from transformer layers via forward hooks.

    Usage:
        cache = ActivationCache(model, layers=[10, 15, 20])
        cache.register_hooks()

        # Run inference (activations are captured automatically)
        with torch.no_grad():
            outputs = model.generate(...)

        # Retrieve activations — dict of {layer_idx: tensor}
        activations = cache.get_activations()
        cache.clear()

        # When done, remove hooks
        cache.remove_hooks()
    """

    def __init__(self, model: PreTrainedModel, layers: Optional[list[int]] = None):
        """Initialize the cache.

        Args:
            model: A HuggingFace causal LM.
            layers: List of layer indices to record. If None, records all layers.
        """
        self.model = model
        self._hooks = []
        self._activations: dict[int, list[torch.Tensor]] = {}

        # Identify the transformer layers in the model.
        # Different architectures use different attribute names.
        self._layer_modules = self._find_layers(model)
        num_layers = len(self._layer_modules)

        if layers is None:
            self.layers = list(range(num_layers))
        else:
            self.layers = [l for l in layers if 0 <= l < num_layers]

        logger.info(
            "ActivationCache initialized for %d/%d layers", len(self.layers), num_layers
        )

    def _find_layers(self, model: PreTrainedModel) -> list:
        """Find the list of transformer layer modules in the model.

        Supports Qwen2, Llama, Gemma 4 multimodal wrapper, and other common
        HuggingFace architectures.
        """
        # Common attribute paths for transformer layers. Try multimodal-nested
        # paths first since they're more specific (Gemma 4 ForConditionalGeneration).
        for attr_path in [
            "model.language_model.layers",  # Gemma 4 multimodal (text decoder nested)
            "model.layers",                  # Qwen2, Llama, most causal LMs
            "transformer.h",                 # GPT-2 family
            "gpt_neox.layers",               # GPT-NeoX
        ]:
            obj = model
            try:
                for attr in attr_path.split("."):
                    obj = getattr(obj, attr)
                return list(obj)
            except AttributeError:
                continue

        raise ValueError(
            f"Cannot find transformer layers in {type(model).__name__}. "
            "Add the layer attribute path to ActivationCache._find_layers()."
        )

    def register_hooks(self):
        """Register forward hooks on the target layers."""
        self.remove_hooks()  # Clear any existing hooks
        self._activations = {layer: [] for layer in self.layers}

        for layer_idx in self.layers:
            module = self._layer_modules[layer_idx]
            hook = module.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(hook)

        logger.debug("Registered %d forward hooks", len(self._hooks))

    def _make_hook(self, layer_idx: int):
        """Create a hook function that captures the hidden state for a layer.

        The hook captures the output hidden state (first element of the layer
        output tuple) and stores it in self._activations.
        """
        def hook_fn(module, input, output):
            # Transformer layer output varies by architecture and version:
            #   - tuple: (hidden_states, ...) — most common
            #   - plain Tensor: hidden_states directly — some newer versions
            #   - BaseModelOutput: has .last_hidden_state attribute
            if isinstance(output, tuple):
                hidden_states = output[0]
            elif isinstance(output, torch.Tensor):
                hidden_states = output
            else:
                hidden_states = output.last_hidden_state

            # Detach and move to CPU to avoid accumulating GPU memory
            self._activations[layer_idx].append(hidden_states.detach().cpu())

        return hook_fn

    def get_activations(
        self,
        token_position: str = "last",
    ) -> dict[int, torch.Tensor]:
        """Retrieve recorded activations.

        Args:
            token_position: Which token's activation to return.
                "last" — activation at the last token position.
                "mean" — mean activation over all token positions.
                "all"  — full sequence of activations (no reduction).

        Returns:
            Dict mapping layer index to activation tensor.
            Shape depends on token_position:
                "last"/"mean": (num_recordings, hidden_size)
                "all": list of (seq_len, hidden_size) tensors
        """
        result = {}
        for layer_idx in self.layers:
            tensors = self._activations[layer_idx]
            if not tensors:
                continue

            if token_position == "all":
                # Return list of per-recording tensors, each (seq_len, hidden_size)
                result[layer_idx] = [t.squeeze(0) for t in tensors]
            elif token_position == "last":
                # Take last token from each recording: (num_recordings, hidden_size)
                result[layer_idx] = torch.stack([t[0, -1, :] for t in tensors])
            elif token_position == "mean":
                # Mean over sequence for each recording: (num_recordings, hidden_size)
                result[layer_idx] = torch.stack([t[0].mean(dim=0) for t in tensors])
            else:
                raise ValueError(f"Unknown token_position: {token_position}")

        return result

    def clear(self):
        """Clear all recorded activations (but keep hooks registered)."""
        self._activations = {layer: [] for layer in self.layers}

    def remove_hooks(self):
        """Remove all forward hooks from the model."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._activations = {}

    @property
    def num_recordings(self) -> int:
        """Number of forward passes recorded since last clear."""
        if not self._activations:
            return 0
        first_layer = next(iter(self._activations))
        return len(self._activations[first_layer])


class RouterCache:
    """Captures MoE router softmax distributions per token per layer.

    For Mixture-of-Experts models, each transformer layer contains a router
    (gate) submodule that scores experts for each token. This cache hooks the
    router output and stores the full post-softmax distribution over experts.

    Storing the full distribution (vs only top-k) enables routing-entropy,
    KL-divergence-between-conditions, and mass-on-unselected-experts analyses
    without re-running inference.

    For dense models, RouterCache is a no-op: _find_router returns None for
    every layer and register_hooks attaches nothing.
    """

    # Submodule paths tried in order to locate the router/gate within a
    # transformer block. First match wins; extend for new architectures.
    _ROUTER_ATTR_CANDIDATES = [
        "router",                    # Gemma 4 (Gemma4TextRouter on the decoder layer)
        "mlp.gate",                  # Mixtral
        "mlp.router",                # alternate naming
        "block_sparse_moe.gate",     # older Mixtral variants
        "feed_forward.gate",         # some implementations
    ]

    def __init__(self, layer_modules: list, layers: list[int]):
        """Initialize the cache.

        Args:
            layer_modules: List of transformer block modules (e.g. ActivationCache._layer_modules).
            layers: Layer indices to record routing for.
        """
        self.layers = layers
        self._hooks: list = []
        self._logits: dict[int, list[torch.Tensor]] = {}
        self._router_modules: dict[int, torch.nn.Module] = {}

        for layer_idx in layers:
            if 0 <= layer_idx < len(layer_modules):
                router = self._find_router(layer_modules[layer_idx])
                if router is not None:
                    self._router_modules[layer_idx] = router

        if self._router_modules:
            logger.info(
                "RouterCache: located router submodule on %d/%d layers",
                len(self._router_modules), len(layers),
            )

    @classmethod
    def _find_router(cls, layer_module) -> Optional[torch.nn.Module]:
        for attr_path in cls._ROUTER_ATTR_CANDIDATES:
            obj = layer_module
            try:
                for attr in attr_path.split("."):
                    obj = getattr(obj, attr)
                return obj
            except AttributeError:
                continue
        return None

    @property
    def has_router(self) -> bool:
        return bool(self._router_modules)

    def register_hooks(self):
        self.remove_hooks()
        self._logits = {layer: [] for layer in self._router_modules}
        for layer_idx, router_module in self._router_modules.items():
            hook = router_module.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # Router output is typically a logits tensor; some implementations
            # return a tuple whose first element is logits.
            logits = output[0] if isinstance(output, tuple) else output
            self._logits[layer_idx].append(logits.detach().cpu())
        return hook_fn

    def clear(self):
        self._logits = {layer: [] for layer in self._router_modules}

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._logits = {}

    def get_distributions(
        self,
        input_len: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> dict[int, torch.Tensor]:
        """Compute softmax over experts and slice to response tokens.

        Args:
            input_len: Number of prompt tokens; response slice starts at this index.
            dtype: Storage dtype for the distributions (bf16 by default).

        Returns:
            Dict mapping layer_idx to tensor of shape (response_len, num_experts).
        """
        result = {}
        for layer_idx, recordings in self._logits.items():
            if not recordings:
                continue
            # Shape: (batch=1, total_seq_len, num_experts) or (total_seq_len, num_experts)
            logits = recordings[0]
            if logits.dim() == 3:
                logits = logits[0]
            response_logits = logits[input_len:]
            # Softmax in fp32 for numerical stability, store in target dtype
            probs = torch.softmax(response_logits.to(torch.float32), dim=-1).to(dtype)
            result[layer_idx] = probs
        return result


def record_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    system_prompt: str,
    layers: list[int],
    max_new_tokens: int = 256,
    token_position: str = "last",
    record_routing: bool = False,
    template_kwargs: Optional[dict] = None,
) -> tuple[dict[int, torch.Tensor], list[str], Optional[list[dict[int, torch.Tensor]]]]:
    """Record activations (and optionally MoE routing) for a list of prompts.

    Uses a two-pass approach for each prompt:
      1. Generate the response with no hooks (to get the response text).
      2. Run a single forward pass over prompt+response with hooks registered
         to capture one clean activation per prompt per layer (and the router
         distribution per response token if record_routing=True).

    This avoids the problem of hooks firing on every autoregressive step during
    generate(), which would produce hundreds of activations per prompt.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        prompts: List of user-message strings (evaluation questions).
        system_prompt: System prompt to set the contrastive condition.
        layers: Which layers to record activations from.
        max_new_tokens: Maximum response length.
        token_position: How to extract a single vector from the response.
            "last" — activation at the last response token.
            "mean" — mean activation over response tokens only.
        record_routing: If True, also capture per-token softmax distributions
            over experts at each layer (only meaningful for MoE models).
        template_kwargs: Optional extra kwargs forwarded to
            tokenizer.apply_chat_template (e.g. {"enable_thinking": False} for
            Gemma 4). Unknown keys are typically passed through to the Jinja
            template environment and ignored if the template doesn't use them.

    Returns:
        (activations, response_texts, routing)
          activations: dict mapping layer index to tensor of shape
              (num_prompts, hidden_size).
          response_texts: list of decoded response strings, length len(prompts).
              Empty string for prompts that produced no response tokens.
          routing: If record_routing=False, returns None. Otherwise, a list of
              length len(prompts) where each element is a dict mapping
              layer_idx to tensor of shape (response_len, num_experts) in bf16.
              For prompts that produced no response, the entry is an empty dict.
    """
    cache = ActivationCache(model, layers=layers)
    layer_modules = cache._layer_modules

    router_cache: Optional[RouterCache] = None
    if record_routing:
        router_cache = RouterCache(layer_modules, layers)
        if not router_cache.has_router:
            logger.warning(
                "record_routing=True but no router submodule found on any layer; "
                "proceeding without routing capture"
            )
            router_cache = None

    device = next(model.parameters()).device
    tmpl_kwargs = template_kwargs or {}

    # Collect one activation per prompt per layer
    per_prompt: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    response_texts: list[str] = []
    per_prompt_routing: list[dict[int, torch.Tensor]] = []

    for i, prompt in enumerate(prompts):
        # Format as chat messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **tmpl_kwargs
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        # --- Pass 1: generate response (no hooks) ---
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy for reproducibility
                pad_token_id=tokenizer.pad_token_id,
            )

        # Full sequence = prompt + response
        full_ids = output_ids[0]  # (total_seq_len,)
        response_len = len(full_ids) - input_len

        if response_len <= 0:
            logger.warning("Prompt %d produced no response tokens, skipping", i)
            response_texts.append("")
            if router_cache is not None:
                per_prompt_routing.append({})
            continue

        # Decode the response text (everything past the prompt)
        response_text = tokenizer.decode(full_ids[input_len:], skip_special_tokens=True)
        response_texts.append(response_text)
        # Log a short preview so cloud runs can be spot-checked from run.log
        # without waiting for the end-of-phase JSON dump.
        logger.info("Prompt %d response[:140]: %r", i, response_text[:140])

        # --- Pass 2: forward pass with hooks to capture activations ---
        cache.register_hooks()
        cache.clear()
        if router_cache is not None:
            router_cache.register_hooks()
            router_cache.clear()

        with torch.no_grad():
            model(full_ids.unsqueeze(0))

        # Extract activations — cache has exactly 1 recording from this pass.
        # The recorded tensor has shape (1, total_seq_len, hidden_size).
        for layer_idx in layers:
            recordings = cache._activations.get(layer_idx, [])
            if not recordings:
                continue
            hidden = recordings[0][0]  # (total_seq_len, hidden_size)

            # Slice to response tokens only
            response_hidden = hidden[input_len:]  # (response_len, hidden_size)

            if token_position == "last":
                per_prompt[layer_idx].append(response_hidden[-1])
            elif token_position == "mean":
                per_prompt[layer_idx].append(response_hidden.mean(dim=0))
            else:
                raise ValueError(f"Unknown token_position: {token_position}")

        if router_cache is not None:
            per_prompt_routing.append(router_cache.get_distributions(input_len))

        cache.remove_hooks()
        if router_cache is not None:
            router_cache.remove_hooks()

        if (i + 1) % 5 == 0 or i == len(prompts) - 1:
            logger.info("Recorded activations for prompt %d/%d", i + 1, len(prompts))

    # Stack into (num_prompts, hidden_size) per layer
    result = {}
    for layer_idx in layers:
        if per_prompt[layer_idx]:
            result[layer_idx] = torch.stack(per_prompt[layer_idx])

    routing_out = per_prompt_routing if router_cache is not None else None
    return result, response_texts, routing_out


def save_activations(activations: dict[int, torch.Tensor], output_dir: Path, prefix: str):
    """Save activation tensors to disk.

    Args:
        activations: Dict of layer_idx -> tensor.
        output_dir: Directory to save to.
        prefix: Filename prefix (e.g., "positive_cat1").
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx, tensor in activations.items():
        path = output_dir / f"{prefix}_layer{layer_idx}.pt"
        torch.save(tensor, path)

    logger.info("Saved activations for %d layers to %s", len(activations), output_dir)


def save_routing(
    routing: list[dict[int, torch.Tensor]],
    output_dir: Path,
    prefix: str,
):
    """Save per-prompt MoE routing distributions to disk.

    Each prompt's routing is a dict mapping layer_idx to a (response_len, num_experts)
    tensor in bf16. Response lengths vary across prompts, so we save one .npz file
    per call with keys "q{i}_l{layer}" pointing to variable-shape ndarrays.

    Args:
        routing: List of per-prompt routing dicts (output of record_activations with
            record_routing=True). Empty dicts (skipped prompts) are still recorded.
        output_dir: Directory to save to.
        prefix: Filename prefix (e.g., "positive_pair_03").
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    for q_idx, per_layer in enumerate(routing):
        for layer_idx, tensor in per_layer.items():
            # bf16 isn't a native numpy dtype; cast to float16 for storage
            arrays[f"q{q_idx}_l{layer_idx}"] = tensor.to(torch.float16).numpy()

    path = output_dir / f"{prefix}.npz"
    np.savez_compressed(path, **arrays)
    logger.info("Saved routing distributions for %d prompts to %s", len(routing), path)


def save_manifest(manifest_rows: list[dict], output_dir: Path, filename: str = "manifest.jsonl"):
    """Save the activation-row manifest as JSONL.

    The manifest gives an explicit row_idx → metadata mapping so future analysis
    can decompose activations along any axis (pair, register, question_type)
    without relying on iteration order in the producing code.

    Args:
        manifest_rows: List of dicts, one per activation row, in the same order
            as the activation tensors. Each row should at least include keys
            row_idx, condition, pair_id, category, register, question_id,
            question_type.
        output_dir: Directory to save to (typically the activations directory).
        filename: Manifest filename (default "manifest.jsonl").
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with open(path, "w") as f:
        for row in manifest_rows:
            f.write(json.dumps(row) + "\n")
    logger.info("Saved manifest with %d rows to %s", len(manifest_rows), path)


def load_activations(output_dir: Path, prefix: str, layers: list[int]) -> dict[int, torch.Tensor]:
    """Load previously saved activation tensors.

    Args:
        output_dir: Directory containing saved activations.
        prefix: Filename prefix used when saving.
        layers: Which layers to load.

    Returns:
        Dict of layer_idx -> tensor.
    """
    output_dir = Path(output_dir)
    result = {}
    for layer_idx in layers:
        path = output_dir / f"{prefix}_layer{layer_idx}.pt"
        if path.exists():
            result[layer_idx] = torch.load(path, weights_only=True)
    return result
