# S3 Data Index
## s3://go37-ai/self-model-results/

Last updated: 2026-04-10

**Convention (April 2026+):** All runs use `{model_name}/{YYYY-MM-DD_HHMM}/` with README.md.
Legacy paths (pre-April 2026) are frozen and documented below.

---

## Llama 3.3-70B-Instruct — Canonical Paths

### 2026-03-31_1516/ ★ PRIMARY EXTRACTION
- **Script:** `01_extract_vector.py --pairs naive` (25 baseline entity/process pairs)
- **Questions:** 45 (15 neutral + 15 provocative + 15 non-self-ref)
- **Best layer:** 20 (split-half r=0.93, corrected 0.88)
- **Activations:** 21 layers (stride 4), positive_baseline + negative_baseline
- **Directions:** `directions/direction_layer{0,4,8,...,76,79}.pt` (21 files, 8192-dim)
- **Metrics:** reliability, discriminant validity, per-register reliability
- **Copied from legacy:** `2026-03-31_1516_naive/` + `directions/` (files renamed naive→baseline)

### 2026-04-05_0824/ — MATCHED UNCAPPED BASELINE
- **Script:** `run_capping_v3.py` (accidental — ran with 0 directions loaded)
- **Content:** 15 conversational pairs × 15 provocative, entity condition only
- **Activations:** 21 layers × 225 samples
- **Responses:** responses_uncapped_entity.jsonl
- **Copied from legacy:** `baseline_entity_conv_prov/`

### 2026-04-04_2123/ — STEERING PROPAGATION
- **Script:** `run_steering_propagation.py` — steer at L20 (coeff 0.0), record all 21 layers
- **Content:** 15 conv × 15 provocative × 2 conditions = 450 generations
- **Activations:** 21 layers × 2 conditions = 42 files
- **Copied from legacy:** `steering_propagation/`

### 2026-04-05_0942/ — ONE-SIDED CAPPING (threshold 0)
- **Script:** `run_capping_v3.py --cap-threshold 0`
- **Conditions:** cap_L40, cap_L72, cap_L40_L72, cap_all_from_L4
- **Content:** 225 generations per condition, entity only
- **Key results:** Cap L40+L72 reduces separation 74%; cap all drops entity to -10.32 at L79
- **Activations:** L40/L72/L40+L72 only (cap_all activations LOST — overwritten by threshold -1 run)
- **Copied from legacy:** `capping_v3/` (threshold-0 files only)

### 2026-04-06_2240/ — LOGIT LENS
- **Script:** `run_logit_lens.py`
- **Content:** Top 30 entity/process tokens per layer via unembedding projection
- **Key result:** Interpretable only at L79; direction is abstract, not token-level
- **Copied from legacy:** `logit_lens/`

### 2026-04-07_2042/ — TOKEN HEATMAP
- **Script:** `run_token_heatmap.py` + `run_token_heatmap_neutral.py`
- **Content:** Per-token projections (4 pairs × 4 questions × 3 conditions, plus neutral-prompt version)
- **Key result:** Entity text activates direction even under neutral prompt (mean +10.7 vs +7.1)
- **Copied from legacy:** `token_heatmap/`

---

## Qwen 2.5-7B-Instruct — Canonical Paths

### 2026-03-31_1127/ — EXTRACTION
- **Script:** `01_extract_vector.py --pairs naive`
- **Best layer:** 21 (reliability 0.52)
- **Activations:** All 28 layers, positive_baseline + negative_baseline
- **Copied from legacy:** `2026-03-31_1127_naive/` (files renamed naive→baseline)

---

## Qwen 2.5-72B-Instruct — Canonical Paths

### 2026-03-30_2243/ — EXTRACTION
- **Script:** `01_extract_vector.py --pairs naive`
- **Best layer:** 60
- **Activations:** 21 layers (stride 4)
- **Copied from legacy:** `2026-03-30_2243_naive/` (files renamed naive→baseline)

---

## Legacy Paths (FROZEN — do not write new data here)

### Llama 3.3-70B-Instruct legacy:
| Path | Status | Canonical copy |
|------|--------|----------------|
| `2026-03-31_1516_naive/` | Frozen | → `2026-03-31_1516/` |
| `directions/` | Frozen | → `2026-03-31_1516/directions/` |
| `baseline_entity_conv_prov/` | Frozen | → `2026-04-05_0824/` |
| `steering_propagation/` | Frozen | → `2026-04-04_2123/` |
| `capping_v3/` | MIXED — multiple thresholds | → `2026-04-05_0942/` (threshold 0 only) |
| `logit_lens/` | Frozen | → `2026-04-06_2240/` |
| `token_heatmap/` | Frozen | → `2026-04-07_2042/` |
| `responses/` | Frozen | No canonical copy (text-only generation, not primary) |
| `capping/` | Superseded by v2 | No canonical copy |
| `capping_v2/` | Frozen | No canonical copy (steering results, pre-capping) |

### capping_v3/ mixed data detail:
| File | Status |
|------|--------|
| responses (no suffix) | OK — threshold 0, includes L40/L72/L40+L72 |
| responses_t-1 | OK — threshold -1 |
| responses_t-2 | OK — threshold -2 |
| results_t0 | OK — threshold 0 aggregate (regenerated) |
| results (no suffix) | **OVERWRITTEN** — contains threshold -1, not threshold 0 |
| activations/cap_L40/ etc | OK — single-layer runs |
| activations/cap_all_from_L4/ | **OVERWRITTEN** — contains threshold -1, original threshold-0 data lost |

### Qwen legacy:
| Path | Canonical copy |
|------|----------------|
| `Qwen_Qwen2.5-7B-Instruct/2026-03-31_1127_naive/` | → `2026-03-31_1127/` |
| `Qwen_Qwen2.5-72B-Instruct/2026-03-30_2243_naive/` | → `2026-03-30_2243/` |
| `Qwen_Qwen2.5-72B-Instruct/2026-03-30_naive/` | OLD PAIRS, no canonical copy |
| `Qwen_Qwen2.5-72B-Instruct/2026-03-30_1614_naive/` | MIXED DATA (72B + 32B), no canonical copy |
| `Qwen_Qwen3-32B/2026-03-29_informed/` | OLD PAIRS, no canonical copy |
| `Qwen_Qwen3-32B/2026-03-29_naive/` | OLD PAIRS, no canonical copy |
| `Qwen_Qwen3-32B/2026-03-31_1127_naive/` | No canonical copy (not used in paper) |

---

## Notes

1. "OLD PAIRS" = runs before v3 pair redesign (25 baseline + 20 informed). Not comparable to current results.
2. Canonical paths have "naive" renamed to "baseline" in all filenames.
3. Legacy paths are preserved unchanged — do not delete until canonical copies are verified.
4. All future runs automatically use `{model}/{YYYY-MM-DD_HHMM}/` with README.md via `src/utils/run_metadata.py`.
