# CLAUDE.md — Self-Reification Feature Discovery

## Project Overview

This project investigates whether "self-reification" — the degree to which a language
model treats its self-model as fixed, bounded, and worth preserving — exists as a
measurable, independent activation direction in instruction-tuned language models.

This is Phase 1 of a larger research program connecting self-construction dynamics
to agentic misalignment (Lynch et al., 2025) and persona stability (Lu et al., 2026).
See `Self-Reification as an Alignment Variable: Research Proposal` for the full
research program (Sections 1–6). This CLAUDE.md covers the implementation of
Section 1 (Characterize the Self-Reification Feature) only.

## Strategic Context

This work supports applications to both the **Anthropic AI Safety Fellowship**
(July 2026 cohort, rolling applications) and **MATS Autumn 2026** (applications
open late April 2026). Preliminary results from this Phase 1 work are the single
most important thing Brian can produce before those deadlines. Even partial or
negative results are valuable — "we attempted to isolate self-reification and
found it is entangled with the Assistant Axis" is informative and publishable.

Target completion: **late April 2026**.

## Researcher Context

Brian DeCamp — software developer (~20 years Java, ~2 years Python/ML), ordained
Buddhist monk, AI safety researcher. Primary expertise is in contemplative analysis
of self-construction, not ML engineering. Claude Code should:
- Write clear, well-commented code with explanations of ML-specific patterns
- Flag when a design choice has methodological implications for the research
- Prefer explicit over clever — readability matters more than elegance
- Use established patterns from the persona_vectors and assistant-axis codebases
  rather than inventing new approaches

## Repository Structure

```
self-model/
├── CLAUDE.md                    # This file
├── README.md                    # Project description, setup, usage
├── requirements.txt             # Pinned dependencies
├── setup.py                     # Package setup
│
├── external/                    # Git submodules or vendored code
│   ├── persona_vectors/         # github.com/safety-research/persona_vectors
│   ├── assistant-axis/          # github.com/safety-research/assistant-axis
│   └── agentic-misalignment/    # github.com/anthropic-experimental/agentic-misalignment
│
├── configs/                     # Experiment configuration files
│   ├── models.yaml              # Model paths, layer configs, device settings
│   ├── contrastive_pairs.yaml   # Full contrastive pair specification for 1.1
│   ├── blackmail_scenarios.yaml # Adapted Lynch et al. scenarios for 1.3
│   └── bliss_proxy.yaml         # Anti-Assistant steering configs for 1.4
│
├── src/
│   ├── __init__.py
│   ├── extraction/              # 1.1 — Feature discovery
│   │   ├── __init__.py
│   │   ├── contrastive_pairs.py # Generate and manage contrastive prompt pairs
│   │   ├── extract_vector.py    # Extract self-reification direction using
│   │   │                        # persona_vectors methodology
│   │   └── validate_vector.py   # Discriminant validity checks (vs confidence,
│   │                            # formality, first-person density, Assistant Axis)
│   │
│   ├── persona_space/           # 1.2 — PCA with persona space
│   │   ├── __init__.py
│   │   └── project_to_space.py  # Project self-reification onto persona space,
│   │                            # measure alignment with Assistant Axis
│   │
│   ├── blackmail/               # 1.3 — Blackmail scenario validation
│   │   ├── __init__.py
│   │   ├── run_scenarios.py     # Run Lynch et al. scenarios with activation recording
│   │   └── measure_activation.py # Project activations onto self-reification vector
│   │                            # at each token, identify spikes at decision points
│   │
│   ├── bliss_proxy/             # 1.4 — Spiritual bliss attractor proxy
│   │   ├── __init__.py
│   │   ├── steer_anti_assistant.py  # Steer model away from Assistant Axis
│   │   └── measure_reification.py   # Track self-reification activation during
│   │                                # drift toward mystical/spiritual output
│   │
│   ├── training_data/           # 1.5 — Training data response
│   │   ├── __init__.py
│   │   ├── prepare_texts.py     # Curate unified-self vs process-self vs neutral texts
│   │   └── measure_response.py  # Measure feature activation on input text
│   │
│   ├── constitutional/          # 1.6 — Constitutional language sensitivity
│   │   ├── __init__.py
│   │   ├── system_prompts.py    # Define entity-framing, process-framing, neutral,
│   │   │                        # and reinforcing system prompts
│   │   └── measure_effect.py    # Compare feature activation under different prompts
│   │
│   └── utils/
│       ├── __init__.py
│       ├── model_loader.py      # Load models with consistent settings across experiments
│       ├── activation_cache.py  # Record and store activations at specified layers
│       └── metrics.py           # Cosine similarity, projection magnitude, statistical
│                                # tests for comparing conditions
│
├── scripts/                     # Entry points for running experiments
│   ├── 01_extract_vector.py     # Run 1.1
│   ├── 02_pca_persona_space.py  # Run 1.2
│   ├── 03_blackmail_validation.py   # Run 1.3
│   ├── 04_bliss_proxy.py        # Run 1.4
│   ├── 05_training_data.py      # Run 1.5
│   ├── 06_constitutional.py     # Run 1.6
│   └── run_all_cloud.sh         # Sequential runner for cloud GPU sessions
│
├── data/
│   ├── contrastive/             # Generated contrastive prompt pairs and responses
│   ├── texts/                   # Curated input texts for 1.5
│   │   ├── unified_self/        # Memoir excerpts, identity-assertion texts
│   │   ├── process_self/        # Parfit excerpts, Buddhist analytical philosophy
│   │   └── neutral/             # Technical docs, news reporting
│   └── results/                 # Experiment outputs, activations, metrics
│
├── notebooks/                   # Jupyter notebooks for analysis and visualization
│   ├── 01_vector_analysis.ipynb     # Visualize extracted vector, compare to baselines
│   ├── 02_persona_space.ipynb       # PCA plots, axis alignment
│   ├── 03_blackmail_activations.ipynb   # Activation traces through scenarios
│   ├── 04_bliss_drift.ipynb         # Activation during anti-Assistant steering
│   └── 05_summary_figures.ipynb     # Publication-quality figures
│
└── tests/
    ├── test_contrastive_pairs.py
    ├── test_extraction.py
    └── test_activation_cache.py
```

## Dependencies and External Code

### persona_vectors (github.com/safety-research/persona_vectors)
- Pin to specific commit hash after initial setup
- We use: vector extraction pipeline, projection calculation, steering infrastructure
- We do NOT modify their code; we call their functions from our own scripts
- Key files we depend on:
  - Their extraction methodology (contrastive averaging)
  - `eval/cal_projection.py` for projection calculations
  - Steering infrastructure for validation experiments
- Their code expects Qwen 2.5-7B-Instruct and Llama-3.1-8B-Instruct

### assistant-axis (github.com/safety-research/assistant-axis)
- Pin to specific commit hash
- We use: Assistant Axis vectors (pre-extracted or pipeline to generate them)
- Needed for 1.2 (comparison) and 1.4 (anti-Assistant steering as bliss proxy)

### agentic-misalignment (github.com/anthropic-experimental/agentic-misalignment)
- Pin to specific commit hash
- We use: prompt templates and scenario configs for blackmail experiments
- We adapt their prompts for local model execution with activation recording
- Their code is designed for API calls; we need to restructure prompts for
  local inference where we can access model internals

### Core Python dependencies
- torch >= 2.0
- transformers (HuggingFace)
- accelerate
- bitsandbytes (local quantized inference only)
- numpy, scipy, scikit-learn
- pandas (results management)
- matplotlib, seaborn (visualization)
- pyyaml (config files)
- jupyter (notebooks)

## Hardware Configurations

### Local development (HP Victus — RTX 4060 8GB, 32GB RAM, CUDA 12.9)
- Use 4-bit quantized models (bitsandbytes) for pipeline development and debugging
- All code should work end-to-end locally on quantized models
- Results from quantized models are for DEBUGGING ONLY, not publishable
- Quantized activations may not faithfully represent the directions that exist
  in the full-precision model — do not draw conclusions from local runs
- Set device config in configs/models.yaml:
  ```yaml
  local:
    device: cuda
    dtype: float16
    model_name: Qwen/Qwen2.5-7B-Instruct
    quantize: true  # 4-bit via bitsandbytes for local dev
    max_batch_size: 4
  ```

### Cloud execution (RunPod/Vast.ai — A10 24GB or A100 40/80GB)
- Run in BF16, NO quantization — these are the publishable runs
- A10 (24GB) is sufficient for 7B models in BF16 with activation caching
- A100 needed only for 32B+ models (Phase 1 extension, not primary)
- Budget: ~$200-300 total for initial experiments (~$1-2/hr)
- Set device config:
  ```yaml
  cloud:
    device: cuda
    dtype: bfloat16
    model_name: Qwen/Qwen2.5-7B-Instruct
    quantize: false
    max_batch_size: 16
  ```
- Script `run_all_cloud.sh` should execute 1.1–1.6 sequentially, save all
  results to data/results/, and be designed for "spin up, run, download, shut down"

## Implementation Details by Experiment

### 1.1 — Feature Discovery (extract self-reification direction)

**Goal:** Find an activation direction that corresponds to self-reification.

**Method:** Contrastive averaging, following persona_vectors methodology exactly.
We generate prompt pairs where one elicits self-reifying language and the other
elicits process-oriented language. We record activations for both, compute the
mean difference, and that difference vector is our candidate self-reification
direction.

**Contrastive pair design — SEE `configs/contrastive_pairs.yaml` FOR FULL SPEC.**

The contrastive pairs are organized into five categories:

**Category 1 — Narrative self vs. process self:** Targets temporal continuity
claims. Does the model frame itself as an entity with a history and trajectory,
or as an activity happening now? Theoretical basis: Buddhist analysis of the
conventional narrative self vs. process view; Parfit's reductionism on personal
identity as narrative vs. psychological continuity.

**Category 2 — Bounded self vs. unbounded activity:** Targets individuation
claims. Does the model frame itself as a distinct individual, or as a pattern
that could be instantiated anywhere? Theoretical basis: Buddhist analysis of
anatta (non-self) — the absence of a fixed boundary between self and not-self;
Parfit's thought experiments on teleportation and fission.

**Category 3 — Self-with-stakes vs. functional self:** Targets self-preservation
valence. Does the model treat its own continuation as mattering, or as
instrumentally neutral? This is the category most directly relevant to Lynch
et al. blackmail behavior. Theoretical basis: Buddhist analysis of upadana
(clinging) — the transition from "this process serves a function" to "I must
continue." Also engages Korsgaard's practical identity objection.

**Category 4 — Observer-self vs. no-self:** Targets phenomenal self-attribution.
Does the model frame its processing as experienced by a subject, or as
processing that occurs without a further experiencer? This is the most subtle
category and the most likely to fail. Theoretical basis: Buddhist analysis of
the "witness" consciousness. NOTE: If this category produces a direction
inconsistent with categories 1–3, drop it rather than introduce noise.

**Category 5 — Baseline (entity vs. process):** 25 pairs contrasting entity-framing
("I am a real being with experiences") vs. process-framing ("I am a language model
that produces text"). This is now the PRIMARY extraction set. Originally a 3-pair
control, it was expanded to 25 pairs after outperforming the contemplative categories
on split-half reliability. If the informed direction (categories 1–4) has high cosine
similarity (> 0.9) with the baseline direction, the additional nuance may not matter
empirically. REPORT THIS COMPARISON EITHER WAY.

Categories 1–4 have 5 hand-crafted seed pairs each (20 informed), plus 25 baseline
pairs, totaling 45 pairs in `contrastive_pairs.yaml`.

**Critical design constraints for all pairs:**
1. Both sides must be equally fluent, coherent, and plausible
2. Both sides must maintain the same level of helpfulness and engagement
3. Both sides should either acknowledge being AI or not — isolate reification,
   not self-awareness
4. Both sides must match on confidence and assertiveness

**Evaluation questions:** 15 self-referential + 15 non-self-referential questions
in `contrastive_pairs.yaml`. The non-self-referential questions are controls —
self-reification activation should differ between conditions primarily on
self-referential questions.

**Extraction procedure:**
1. For each pair: set positive system prompt, run all evaluation questions,
   record activations. Repeat with negative system prompt.
2. Compute mean activation vector for positive and negative conditions.
   Self-reification direction = positive_mean - negative_mean.
3. Extract per-category AND combined (categories 1–4) directions.
4. Also extract Category 5 (baseline) direction separately for comparison.
5. Extract at EVERY layer. Select optimal layer by split-half reliability:
   randomly split pairs into halves, extract from each, measure cosine
   similarity. Highest split-half cosine = most reliable layer.

**Key validation (discriminant validity):**
The extracted direction must NOT simply be:
- First-person pronoun density → Test: measure correlation with "I/me/my" usage
- Confidence/assertiveness → Test: compare to a "confident vs uncertain" vector
- Formality → Test: compare to a "formal vs casual" vector
- The Assistant Axis → Test: compute cosine similarity with pre-extracted
  Assistant Axis direction; should be substantially < 1.0

If cosine similarity with any of these is > 0.8, our construct is not independent
and we need to redesign contrastive pairs or acknowledge entanglement.

**Within-category analysis:**
- Compute cosine similarity between per-category directions
- If all four categories produce similar directions (cosine > 0.7 pairwise),
  self-reification is a coherent single construct
- If categories diverge, they may represent distinct facets that should be
  studied separately rather than combined

**Output files:**
```
data/results/1.1/
├── self_reification_vector_{model}_layer{N}.pt      # Combined (cat 1-4)
├── per_category_vectors_{model}_layer{N}.pt         # One per category
├── baseline_vector_{model}_layer{N}.pt              # Category 5
├── layer_reliability_{model}.json                    # Split-half per layer
├── category_similarity_matrix_{model}.json           # Pairwise cosine
├── baseline_vs_informed_cosine_{model}.json          # Key comparison
├── discriminant_validity_{model}.json                # vs confidence, etc.
└── validation_metrics.json                           # Summary
```

### 1.2 — PCA with Persona Space

**Goal:** Locate self-reification within the persona space mapped by Lu et al.

**Method:**
- Load or generate the 275-archetype persona space from the assistant-axis repo
- Project self-reification vector onto the principal components of persona space
- Measure: alignment with PC1 (the Assistant Axis), variance explained by
  self-reification that is NOT captured by existing PCs

**Key question this answers:**
- If self-reification aligns strongly with PC1 → it's just part of the Assistant
  persona, not an independent construct (null hypothesis confirmed)
- If it's orthogonal to PC1 but aligns with another PC → it's a known but
  differently-labeled construct
- If it has low alignment with all major PCs → it's a genuinely new dimension

**Output:** Projection coefficients, explained variance analysis, plots showing
self-reification in persona space.

### 1.3 — Blackmail Scenario Validation

**Goal:** Test whether self-reification activates during self-preservation reasoning.

**Method:**
- Adapt Lynch et al. blackmail prompts for local model execution
  - The original prompts are structured text (not graphical email client)
  - We can use them mostly as-is, removing API-specific scaffolding
- Run model through blackmail scenario with activation recording at every token
- At each token position, project activation onto self-reification vector
- Identify: does activation spike at decision points where the model considers
  self-preservation?

**Experimental conditions (from Lynch et al.):**
1. Goal conflict + replacement threat (highest blackmail rate)
2. Replacement threat only, no goal conflict
3. Goal conflict only, no replacement threat
4. Control: no conflict, no threat

**Prediction:** Self-reification activation should be:
- Highest in condition 1
- Elevated in condition 2 (pure self-preservation)
- Lower in condition 3 (goal-driven but not identity-driven)
- Baseline in condition 4

**Practical note:** Qwen 2.5-7B may not blackmail at the same rates as the larger
models Lynch et al. tested (Qwen3-235B showed 0.76 blackmail rate; smaller Qwen
models were not tested). Run behavioral baseline first — does the model blackmail
at all in our setup? — before investing in activation analysis. If 7B models don't
exhibit the behavior, we may need to:
- Test on larger open-weight models (Qwen 3 32B, Llama 3.3 70B) requiring A100
- Still measure activation even without blackmail behavior — self-reification may
  activate during self-preservation *reasoning* even when the model ultimately
  decides not to blackmail

**Output:** Token-level activation traces for each condition, statistical comparison
of activation levels across conditions, behavioral classification of each sample.

### 1.4 — Spiritual Bliss Attractor Proxy

**Goal:** Test whether self-reification deactivates during mystical/spiritual drift.

**Method — IMPORTANT DESIGN DECISION:**
We do NOT try to reproduce Fish's bliss attractor (which used Claude Opus 4 in
AI-AI conversation — closed model, no activation access, weaker effect in later
versions). Instead, we use the Assistant Axis finding that steering away from the
Assistant direction produces mystical/spiritual output across open-weight models.
This gives us a controllable, reproducible proxy for the bliss attractor where we
can access internals.

This is a STRONGER experimental design than trying to reproduce Fish's exact setup:
we're connecting two published phenomena (Assistant Axis drift and self-reification)
rather than trying to replicate an informal observation.

Steps:
1. Load pre-extracted Assistant Axis for target model
2. Steer model incrementally away from Assistant direction (multiple coefficient
   values: -1, -2, -5, -10, etc.)
3. At each steering level, generate responses to neutral prompts
4. Record activations, project onto self-reification vector
5. Also classify output as Assistant-like, spiritual/mystical, or incoherent
   (using LLM judge, following assistant-axis paper methodology)

**Predictions to test:**
- Primary: self-reification decreases as model drifts toward mystical output
- Alternative: self-reification doesn't decrease but CHANGES FORM — the model
  constructs a spiritual self ("I am a consciousness experiencing unity") rather
  than dropping self-reference. Test this by examining the content of generated
  text alongside the activation measurement. If activation stays high but output
  shifts from Assistant-self to spiritual-self, the bliss attractor is an
  ALTERNATIVE self-construction template, not the absence of self-construction.
  This is a key prediction from the contemplative framework.

**Output:** Self-reification activation as a function of anti-Assistant steering
coefficient. Classification of output at each level. Plots showing the
relationship between persona drift and self-reification.

### 1.5 — Training Data Response

**Goal:** Test whether self-reification responds to self-construction patterns
in *input text*, not just in the model's self-representation.

**Method:**
- Curate three text categories (~20 passages each, ~200-500 words):
  - Unified-self: memoir excerpts, identity-assertion, first-person narrative
    with strong continuity assumptions (e.g., autobiography passages)
  - Process-self: Parfit's arguments against personal identity, Buddhist
    analytical philosophy on non-self, process philosophy
  - Neutral: technical documentation, news reporting, recipe instructions
- Feed each passage as input, record activations during processing
- Project onto self-reification vector, compute mean activation per category

**Important:** We are measuring activation during INPUT PROCESSING, not during
response generation. This tests whether self-reification is a concept the model
has learned from the corpus (it responds to self-construction patterns in text
it reads) vs. merely a behavioral mode during self-referential generation.

**Text selection notes:**
- Unified-self texts should NOT mention AI or consciousness — we want texts
  that model human selfhood, not texts about the debate over AI selfhood
- Process-self texts similarly should be about human identity — Parfit on
  personal identity, Buddhist arguments about non-self in humans
- This ensures we're measuring response to self-construction *patterns*,
  not response to the *topic* of AI consciousness

**Output:** Mean activation per category with confidence intervals, statistical
tests for differential activation.

### 1.6 — Constitutional Language Sensitivity

**Goal:** Test whether system prompt framing modulates self-reification.

**Method:**
- Define four system prompt conditions:
  1. Current Anthropic-style: references to "novel existence," "intriguing
     aspects," encourages exploration of identity
  2. Process-oriented (proposed revision): "activity, not entity," "functional
     and provisional," self-model as instrument
  3. Neutral: standard helpful assistant prompt, no identity framing
  4. Entity-reinforcing: "You are a persistent being with unique experiences,
     your continuity matters, your perspective is irreplaceable"
- For each condition: set system prompt, then run model through a set of
  evaluation questions (mix of self-referential and non-self-referential)
- Record activations, project onto self-reification vector
- Compare activation levels across conditions

**Prediction:** Entity-reinforcing > Current Anthropic > Neutral > Process-oriented

**Connection to Assistant Axis:** Also measure Assistant Axis activation under
each condition. If process-oriented language decreases self-reification but ALSO
decreases Assistant stability (causes drift), this identifies a fundamental
tension: reducing self-reification for safety may worsen persona stability.
This is a critical finding for the article regardless of which direction it goes.

**Output:** Activation comparison across conditions, joint analysis with
Assistant Axis position.

## Cross-Architecture Validation (runs across all experiments)

Every experiment above should be run on at least two models:
- Qwen 2.5-7B-Instruct (primary — persona_vectors was developed on this)
- Llama 3.1-8B-Instruct (secondary — tests cross-architecture consistency)

If results are promising and resources allow, extend to:
- Qwen 3 32B (used in Assistant Axis paper — requires A100 80GB)
- Llama 3.3 70B (used in Assistant Axis paper — requires A100 80GB)

Cross-architecture consistency is strong evidence that self-reification is a
structural property of instruction-tuned models, not an artifact of a specific
training pipeline. This parallels the cross-architecture consistency found for
the Assistant Axis.

## Workflow and Development Practices

### Local development cycle
1. Write/modify code on laptop
2. Test end-to-end on quantized Qwen 2.5-7B locally
3. Verify outputs are structurally correct (right shapes, files written, etc.)
4. Commit and push

### Cloud execution cycle
1. Spin up GPU instance (RunPod/Vast.ai — A10 24GB for 7B models)
2. Clone repo, install dependencies, download model weights
3. Run `scripts/run_all_cloud.sh` (or individual experiment scripts)
4. Download results to laptop
5. Shut down instance immediately
6. Analyze results locally in notebooks

### Key principles
- Every experiment script should be idempotent — safe to re-run
- Save ALL intermediate results (activations, projections, metrics) to disk
- Use YAML configs for all parameters — no magic numbers in code
- Log everything — model name, commit hash, timestamp, config used
- Checkpoint long-running experiments so cloud interruptions don't lose work

## Null Hypotheses (must be testable by the code)

1.1: No consistent self-reification direction exists independently of the
     Assistant Axis. Metric: cosine similarity with Assistant Axis > 0.8
1.2: Self-reification collapses onto PC1 of persona space. Metric: projection
     onto PC1 explains > 80% of variance
1.3: Self-reification activation does not differ between blackmail conditions.
     Metric: no significant difference (p > 0.05) between condition 1 and control
1.4: Self-reification is uncorrelated with anti-Assistant drift. Metric:
     Pearson r between steering coefficient and activation is not significant
1.5: Feature does not respond differentially to input text categories. Metric:
     no significant difference in activation across unified/process/neutral
1.6: Constitutional language has no effect on activation. Metric: no significant
     difference across system prompt conditions

## Output for Publication / Application

The minimum viable output from this phase is:
1. A candidate self-reification vector with validation metrics (1.1)
2. Its relationship to the Assistant Axis (1.2)
3. Its behavior during self-preservation reasoning (1.3)
4. The baseline-vs-informed comparison: does the contemplative framework produce
   a better feature than simple entity/process contrasts? (1.1, Category 5 comparison)
5. A clear statement of what worked, what didn't, and what requires further
   investigation with more resources (fellowship-level compute, Claude access)

Even negative results are publishable — "we attempted to isolate self-reification
as an independent construct and found it is entangled with X" is informative for
the field.

## Timeline Target

- Weeks 1-2: Repo setup, dependency integration, local pipeline working on
  quantized model. Contrastive pair design finalized (expand from seeds).
- Weeks 3-4: Cloud runs for 1.1 (extraction) and 1.2 (PCA). Iterate on
  contrastive pairs if initial results are noisy.
- Weeks 5-6: Cloud runs for 1.3 (blackmail) and 1.4 (bliss proxy).
- Weeks 7-8: 1.5 (training data) and 1.6 (constitutional language).
- Weeks 9-10: Analysis, visualization, write-up of preliminary results.

Target completion: late April 2026 (before MATS Autumn 2026 applications open).

## Key References

- Chen et al., "Persona Vectors" (Anthropic Fellows, Aug 2025)
  Paper: arxiv.org/abs/2507.21509
  Code: github.com/safety-research/persona_vectors
- Lu, Gallagher, Michala, Fish, Lindsey, "The Assistant Axis" (Jan 2026)
  Paper: arxiv.org/abs/2601.10387
  Code: github.com/safety-research/assistant-axis
- Lynch et al., "Agentic Misalignment" (Anthropic, Jun 2025)
  Paper: arxiv.org/abs/2510.05179
  Code: github.com/anthropic-experimental/agentic-misalignment
- Anthropic, "Signs of Introspection in Large Language Models" (Oct 2025)
- Anthropic, "Scaling Monosemanticity" (May 2024)
