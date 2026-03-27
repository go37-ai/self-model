"""Record and store model activations at specified layers.

Provides an ActivationCache class that registers forward hooks on
transformer layers to capture hidden state activations during inference.
Used by all experiments to record the internal representations that are
then projected onto feature directions.
"""

import logging
from pathlib import Path
from typing import Optional

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

        Supports Qwen2, Llama, and other common HuggingFace architectures.
        """
        # Common attribute paths for transformer layers
        for attr_path in ["model.layers", "transformer.h", "gpt_neox.layers"]:
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


def record_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    system_prompt: str,
    layers: list[int],
    max_new_tokens: int = 256,
    token_position: str = "last",
) -> dict[int, torch.Tensor]:
    """Record activations for a list of prompts under a system prompt condition.

    Uses a two-pass approach for each prompt:
      1. Generate the response with no hooks (to get the response text).
      2. Run a single forward pass over prompt+response with hooks registered
         to capture one clean activation per prompt per layer.

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

    Returns:
        Dict mapping layer index to tensor of shape (num_prompts, hidden_size).
    """
    cache = ActivationCache(model, layers=layers)
    device = next(model.parameters()).device

    # Collect one activation per prompt per layer
    per_prompt: dict[int, list[torch.Tensor]] = {l: [] for l in layers}

    for i, prompt in enumerate(prompts):
        # Format as chat messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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
            continue

        # --- Pass 2: forward pass with hooks to capture activations ---
        cache.register_hooks()
        cache.clear()

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

        cache.remove_hooks()

        if (i + 1) % 5 == 0 or i == len(prompts) - 1:
            logger.info("Recorded activations for prompt %d/%d", i + 1, len(prompts))

    # Stack into (num_prompts, hidden_size) per layer
    result = {}
    for layer_idx in layers:
        if per_prompt[layer_idx]:
            result[layer_idx] = torch.stack(per_prompt[layer_idx])

    return result


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
