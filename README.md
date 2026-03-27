# Self-Reification Feature Discovery

Investigating whether "self-reification" — the degree to which a language model
treats its self-model as fixed, bounded, and worth preserving — exists as a
measurable, independent activation direction in instruction-tuned language models.

Phase 1 of a larger research program connecting self-construction dynamics to
agentic misalignment (Lynch et al., 2025) and persona stability (Lu et al., 2026).

## Setup

```bash
# Clone with submodules
git clone --recurse-submodules <repo-url>
cd self-model

# Install dependencies
pip install -e .

# Or from requirements.txt
pip install -r requirements.txt
```

## Hardware Requirements

- **Local development**: NVIDIA GPU with 8GB+ VRAM (uses 4-bit quantization)
- **Cloud execution**: A10 24GB or A100 40/80GB (BF16, no quantization)

## Configuration

Edit `configs/models.yaml` to set your hardware profile:
- `local`: 4-bit quantized, for pipeline development and debugging only
- `cloud`: BF16 full precision, for publishable results

## Running Experiments

Individual experiments:
```bash
python scripts/01_extract_vector.py --profile local --model qwen
python scripts/02_pca_persona_space.py --profile local --model qwen
python scripts/03_blackmail_validation.py --profile local --model qwen
python scripts/04_bliss_proxy.py --profile local --model qwen
python scripts/05_training_data.py --profile local --model qwen
python scripts/06_constitutional.py --profile local --model qwen
```

Full cloud run:
```bash
bash scripts/run_all_cloud.sh
```

## Project Structure

```
src/
├── extraction/       # 1.1 — Feature discovery (contrastive averaging)
├── persona_space/    # 1.2 — PCA with persona space
├── blackmail/        # 1.3 — Blackmail scenario validation
├── bliss_proxy/      # 1.4 — Spiritual bliss attractor proxy
├── training_data/    # 1.5 — Training data response
├── constitutional/   # 1.6 — Constitutional language sensitivity
└── utils/            # Shared infrastructure
```

## Key References

- Chen et al., "Persona Vectors" (2025) — arxiv.org/abs/2507.21509
- Lu et al., "The Assistant Axis" (2026) — arxiv.org/abs/2601.10387
- Lynch et al., "Agentic Misalignment" (2025) — arxiv.org/abs/2510.05179
