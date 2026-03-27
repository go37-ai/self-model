"""Load models with consistent settings across all experiments.

Handles both local (4-bit quantized) and cloud (BF16) configurations,
reading settings from configs/models.yaml. All experiments should use
this module rather than loading models directly.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Default config path relative to project root
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "models.yaml"


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load the models.yaml configuration file.

    Args:
        config_path: Path to models.yaml. Defaults to configs/models.yaml.

    Returns:
        Parsed YAML config as a dictionary.
    """
    config_path = config_path or DEFAULT_CONFIG_PATH
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_model_config(
    model_key: str = "qwen",
    profile: str = "local",
    config_path: Optional[Path] = None,
) -> dict:
    """Build a combined config dict from profile + model settings.

    Args:
        model_key: Which model to use ("qwen" or "llama").
        profile: Hardware profile ("local" or "cloud").
        config_path: Path to models.yaml.

    Returns:
        Dict with keys: name, num_layers, hidden_size, device, dtype,
        quantize, max_batch_size.
    """
    config = load_config(config_path)
    profile_cfg = config["profiles"][profile]
    model_cfg = config["models"][model_key]
    return {**model_cfg, **profile_cfg}


def load_model_and_tokenizer(
    model_key: str = "qwen",
    profile: str = "local",
    config_path: Optional[Path] = None,
):
    """Load a model and tokenizer using settings from config.

    For local profile: loads in 4-bit quantization via bitsandbytes.
    For cloud profile: loads in BF16 without quantization.

    Args:
        model_key: Which model to use ("qwen" or "llama").
        profile: Hardware profile ("local" or "cloud").
        config_path: Path to models.yaml.

    Returns:
        Tuple of (model, tokenizer, model_config_dict).
    """
    cfg = get_model_config(model_key, profile, config_path)
    model_name = cfg["name"]
    device = cfg["device"]

    # Map string dtype to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[cfg["dtype"]]

    logger.info(
        "Loading model=%s profile=%s dtype=%s quantize=%s",
        model_name,
        profile,
        cfg["dtype"],
        cfg["quantize"],
    )

    # Build model loading kwargs
    model_kwargs = {
        "dtype": torch_dtype,
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if cfg["quantize"]:
        # 4-bit quantization for local development
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Ensure pad token is set (some models don't have one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    device_map = getattr(model, "hf_device_map", "N/A")
    logger.info("Model loaded successfully. Device map: %s", device_map)

    return model, tokenizer, cfg
