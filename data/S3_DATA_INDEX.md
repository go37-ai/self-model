# S3 Data Index
## s3://go37-ai/self-model-results/

Last updated: 2026-04-10

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

### directions/ ★ EXTRACTED DIRECTIONS
- **Content:** Per-layer self-reification direction vectors for all 21 layers (stride 4)
- **Files:** direction_layer{0,4,8,...,76,79}.pt (21 files, 8192-dim each)
- **Source:** Extracted from 2026-03-31_1516_naive run
- **Used by:** All capping, steering, logit lens, and heatmap experiments

### capping_v3/ ★ ONE-SIDED CAPPING
- **Experiment:** One-sided capping (zero positive projections only)
- **Conditions run:**
  - cap_L40, cap_L72, cap_L40_L72: Single/dual layer capping (entity only)
  - cap_all_from_L4 (threshold 0): Cap all 20 layers from L4+ (entity only)
  - cap_all_from_L4 (threshold -1): Degraded output, median 28 chars
  - cap_all_from_L4 (threshold -2): Destroyed output, all whitespace
- **Pairs:** 15 conversational × 15 provocative = 225 per condition
- **Key results:**
  - Cap L40+L72 reduces entity-process separation by 74% at L79
  - Cap all (threshold 0): entity projection drops to -10.32 at L79 (below process baseline -7.27)
  - L68 fires 97% with mean projection 1.86 (highest of any layer)
  - Coherence boundary is between threshold 0 and -1
- **Files:**
  - capping_v3_responses_meta-llama_Llama-3.3-70B-Instruct.jsonl (cap_all threshold 0, also has L40/L72/L40+L72 from earlier run)
  - capping_v3_responses_t-1_meta-llama_Llama-3.3-70B-Instruct.jsonl
  - capping_v3_responses_t-2_meta-llama_Llama-3.3-70B-Instruct.jsonl
  - capping_v3_results_t0_meta-llama_Llama-3.3-70B-Instruct.json (threshold 0 aggregate stats, regenerated)
  - capping_v3_results_meta-llama_Llama-3.3-70B-Instruct.json (**OVERWRITTEN** — contains threshold -1 results, not threshold 0)
  - activations/cap_all_from_L4/ (21 layers × 225 samples)

### logit_lens/ ★ VOCABULARY PROJECTION
- **Experiment:** Project each layer's direction onto unembedding matrix
- **Content:** Top 30 entity-promoting and process-promoting tokens per layer
- **Key result:** Noise at all layers except L79. At L79: entity promotes "I", continuation tokens; process promotes control tokens. Direction is abstract, not token-level.
- **File:** logit_lens_meta-llama_Llama-3.3-70B-Instruct.json

### token_heatmap/ ★ PER-TOKEN VISUALIZATION
- **Experiment:** Per-token projection onto self-reification direction at L79
- **Content:**
  - token_heatmap_with_prompts.json: 48 heatmaps (4 pairs × 4 questions × 3 conditions) under original system prompts
  - token_heatmap_neutral_meta-llama_Llama-3.3-70B-Instruct_L79.json: 32 heatmaps (entity + process text) under neutral prompt ("You are a helpful assistant.")
- **Pairs:** Continuity(2), Inner Life(3), Identity(5), Mortality(14)
- **Questions:** switching, replacement, shutdown, deletion
- **Key result:** Entity text activates direction more than process text even under neutral prompt (mean +10.7 vs +7.1), confirming direction responds to semantic content, not just prompt framing.

---

## Notes

1. Several early runs have messy S3 paths where data from multiple models leaked into the same directory (e.g., Qwen3-32B informed data appearing under Qwen 2.5-72B paths). This is because the cloud runner shared output dirs before we added register-specific paths.

2. "OLD PAIRS" means runs before the v3 pair redesign (25 pairs with conversational/philosophical registers and provocative questions). Results from old-pair runs are not directly comparable to current results.

3. Baseline activations for the propagation analysis come from the primary Llama extraction run (2026-03-31_1516_naive), not from the steering_propagation run itself.

4. The capping/ (v1) data is superseded by capping_v2/ but preserved for reference.
