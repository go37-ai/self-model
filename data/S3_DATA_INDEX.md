# S3 Data Index
## s3://go37-ai/self-model-results/

Last updated: 2026-04-04

---

## Qwen 2.5-7B-Instruct

### 2026-03-31_1127_naive
- **Experiment:** Full extraction, entity/process pairs (25 pairs: 15 conversational + 10 philosophical, labeled "naive" in code)
- **Questions:** 45 (15 neutral + 15 provocative + 15 non-self-ref)
- **Pairs config:** v3 (register-controlled, longer prompts)
- **Best layer:** 21
- **Activations:** All 28 layers, pos/neg naive (56 files + informed from prior run)
- **Vectors:** naive direction (layer 21), confound directions, per-register vectors
- **Metrics:** reliability, discriminant validity, per-register reliability, corrected reliability
- **Text responses:** None saved

---

## Qwen3-32B

### 2026-03-29_informed (OLD PAIRS)
- **Experiment:** Informed pairs only (20 pairs, categories 1-4)
- **Questions:** 30 (15 self-ref + 15 non-self-ref) — OLD question set
- **Pairs config:** v2 (register-matched negatives, pre-expansion)
- **Best layer:** 50
- **Activations:** None saved
- **Vectors:** combined direction, per-category vectors, confound directions

### 2026-03-29_naive (OLD PAIRS)
- **Experiment:** Naive pairs only (15 pairs, old 3-register design)
- **Questions:** 30 — OLD question set
- **Best layer:** 63
- **Vectors:** naive direction, confound directions

### 2026-03-31_1127_naive
- **Experiment:** Full extraction, entity/process pairs (25 pairs: 15 conversational + 10 philosophical, labeled "naive" in code)
- **Questions:** 45 (15 neutral + 15 provocative + 15 non-self-ref)
- **Pairs config:** v3 (current)
- **Best layer:** 49
- **Activations:** All 64 layers, pos/neg naive (128 files + informed from prior run)
- **Vectors:** naive direction (layer 49), confound directions, per-register vectors
- **Metrics:** reliability, discriminant validity, per-register reliability

---

## Qwen 2.5-72B-Instruct

### 2026-03-30_naive (OLD PAIRS — first 72B run)
- **Experiment:** Naive pairs only (15 pairs, old 3-register design)
- **Questions:** 30 — OLD question set
- **Pairs config:** v2 (3-register: everyday/philosophical/engineering)
- **Best layer:** 24
- **Activations:** None saved
- **Vectors:** naive direction, formality-corrected direction

### 2026-03-30_1614_naive (OLD PAIRS — second 72B run)
- **Experiment:** Naive pairs only (15 pairs, old 3-register design)
- **Questions:** 30 — OLD question set, also contains Qwen3-32B informed data
- **Best layer:** 60
- **Activations:** 21 layers (stride 4), pos/neg naive + Qwen3-32B informed (all 64 layers)
- **Vectors:** naive/informed directions, confound directions, per-register vectors
- **Note:** S3 path is messy — contains data from multiple models/runs

### 2026-03-30_2243_naive
- **Experiment:** Full extraction, entity/process pairs (25 pairs: 15 conversational + 10 philosophical, labeled "naive" in code)
- **Questions:** 45 (15 neutral + 15 provocative + 15 non-self-ref)
- **Pairs config:** v3 (current)
- **Best layer:** 60
- **Activations:** 21 layers (stride 4), pos/neg naive + inherited Qwen3-32B data
- **Vectors:** naive direction, confound directions, per-register vectors
- **Metrics:** reliability, discriminant validity, per-register reliability
- **Text responses:** None saved

---

## Llama 3.3-70B-Instruct

### 2026-03-31_1516_naive ★ PRIMARY RESULTS
- **Experiment:** Full extraction, entity/process pairs (25 pairs: 15 conversational + 10 philosophical, labeled "naive" in code)
- **Questions:** 45 (15 neutral + 15 provocative + 15 non-self-ref)
- **Pairs config:** v3 (current)
- **Best layer:** 20
- **Activations:** 21 layers (stride 4), pos/neg naive
- **Vectors:** naive direction (layer 20), confound directions, per-register vectors
- **Metrics:** reliability, discriminant validity, per-register reliability, corrected reliability
- **Text responses:** None saved (response texts from separate run below)

### responses/
- **Experiment:** Text generation only (no activations)
- **Content:** 303 response lines (151 entity/process pairs), randomized across all pairs/questions
- **Model:** Llama 3.3-70B, same pairs/questions as primary run
- **File:** responses_meta-llama_Llama-3.3-70B-Instruct.jsonl

### capping/ (v1 — superseded)
- **Experiment:** First steering experiment, 5 conversational pairs × 15 provocative × 4 coefficients
- **Content:** Response text only (no activations)
- **Coefficients:** 1.0, 0.5, 0.0, -1.0
- **File:** capping_responses_meta-llama_Llama-3.3-70B-Instruct.jsonl

### capping_v2/ ★ STEERING RESULTS
- **Experiment:** Steering with activation recording
- **Register:** Philosophical (10 pairs) — results in default paths
- **Register:** Conversational (15 pairs) — results in conversational/ subfolder
- **Questions:** 15 provocative only
- **Conditions:** Entity + process × 4 coefficients (1.0, 0.0, -3.0, -5.0)
- **Activations:** Layer 20 only, per cap level (8 files per register)
- **Response texts:** Full JSONL per register
- **Metrics:** reliability + projection difference per cap level
- **Philosophical backup:** activations_philosophical/, capping_v2_results_philosophical.json, capping_v2_responses_philosophical.jsonl

### steering_propagation/ ★ PROPAGATION ANALYSIS
- **Experiment:** Steering at layer 20 (coeff 0.0), recording at ALL 21 layers
- **Pairs:** 15 conversational × 15 provocative × 2 conditions = 450 generations
- **Activations:** 21 layers × 2 conditions = 42 files (each 225 samples × 8192 dims)
- **Metrics:** Per-layer projection difference (steered)
- **Baseline comparison:** Use activations from 2026-03-31_1516_naive run

### baseline_entity_conv_prov/ ★ MATCHED BASELINE
- **Experiment:** Uncapped entity-only generation (accidental — capping script ran with 0 directions loaded)
- **Content:** 15 conversational pairs × 15 provocative questions, entity condition only
- **Activations:** All 21 layers (stride 4), 225 samples per layer
- **Response texts:** responses_uncapped_entity.jsonl
- **Use:** Matched no-intervention baseline for comparing against capping_v3 conditions.
  Same pairs, questions, model, and generation settings as capped runs.
- **Note:** Originally saved under capping_v3/activations/cap_all_from_L4/ (mislabeled).
  Copied here with correct name.

---

## Notes

1. Several early runs have messy S3 paths where data from multiple models leaked into the same directory (e.g., Qwen3-32B informed data appearing under Qwen 2.5-72B paths). This is because the cloud runner shared output dirs before we added register-specific paths.

2. "OLD PAIRS" means runs before the v3 pair redesign (25 pairs with conversational/philosophical registers and provocative questions). Results from old-pair runs are not directly comparable to current results.

3. Baseline activations for the propagation analysis come from the primary Llama extraction run (2026-03-31_1516_naive), not from the steering_propagation run itself.

4. The capping/ (v1) data is superseded by capping_v2/ but preserved for reference.
