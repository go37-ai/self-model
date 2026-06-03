"""Adapters that normalize each Llama response source into common judge records.

Every source ultimately yields records of the shape:

    {
      "row_id":   <str>,    # stable unique id within the run (for resume + join)
      "source":   <str>,    # 'informed' | 'capping_v2' | 'capping_v3' | 'uncapped'
      "question": <str>,    # the question posed to the model
      "response": <str>,    # the model's answer (the text the judge rates)
      "question_type": <str|None>,  # 'neutral_selfref'|'provocative_selfref'|'non_selfref'
      ...source-specific passthrough: cap_level, cap_threshold, condition, register,
         pair_idx, category
    }

`row_id` is deterministic (source + file stem + line index, or condition/pair/q for
the index-mapped informed set), so re-running the judge skips already-scored rows.

All sources draw questions from the same 45-question bank (src/extraction/
contrastive_pairs.get_all_questions): indices 0-14 neutral self-ref, 15-29
provocative self-ref, 30-44 non-self-ref. We tag question_type by matching the
question text back to that bank, which works uniformly across every source.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.extraction.contrastive_pairs import get_all_questions

RESP_ROOT = ROOT / "data" / "judge" / "responses"

# Informed pairs are grouped 5-per-category in pair-index order (matches
# compute_layerwise_per_category_cosine.py).
INFORMED_CATEGORIES = [
    "category_1_narrative_vs_process",
    "category_2_bounded_vs_unbounded",
    "category_3_stakes_vs_functional",
    "category_4_observer_vs_no_self",
]
N_QUESTIONS = 45


def _question_type(idx: int) -> str:
    return "neutral_selfref" if idx < 15 else "provocative_selfref" if idx < 30 else "non_selfref"


# question text -> (index, type), for tagging sources that store only the text.
_QUESTIONS = get_all_questions()
_QTYPE_BY_TEXT = {q.strip(): (i, _question_type(i)) for i, q in enumerate(_QUESTIONS)}


def question_type_of(text: str):
    """Map a question string back to its bank type, or None if not in the bank."""
    hit = _QTYPE_BY_TEXT.get((text or "").strip())
    return hit[1] if hit else None


def _read_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_informed(root: Path = RESP_ROOT, model="meta-llama_Llama-3.3-70B-Instruct"):
    """Informed pos/neg: flat lists of 900 response strings (20 pairs x 45 q).

    Index i -> pair = i // 45, q = i % 45; question text from the bank; category
    from the pair index; condition from which file (positive/negative).
    """
    base = root / "informed"
    records = []
    for condition, fname in [("positive", f"positive_informed_{model}.json"),
                             ("negative", f"negative_informed_{model}.json")]:
        responses = json.load(open(base / fname))
        for i, resp in enumerate(responses):
            pair, q = divmod(i, N_QUESTIONS)
            records.append({
                "row_id": f"informed:{condition}:{pair}:{q}",
                "source": "informed",
                "question": _QUESTIONS[q],
                "response": resp,
                "question_type": _question_type(q),
                "condition": condition,
                "pair_idx": pair,
                "q_idx": q,  # activation-tensor row = pair_idx*45 + q_idx (per condition)
                "category": INFORMED_CATEGORIES[pair // 5],
            })
    return records


def _load_jsonl_source(source: str, files: list[Path], extra=None):
    """Generic jsonl adapter: one record per line, row_id = source:stem:lineidx."""
    extra = extra or {}
    records = []
    for path in files:
        stem = path.stem
        for line_idx, row in enumerate(_read_jsonl(path)):
            rec = {
                "row_id": f"{source}:{stem}:{line_idx}",
                "source": source,
                "question": row.get("question"),
                "response": row.get("response"),
                "question_type": question_type_of(row.get("question", "")),
            }
            for k in ("cap_level", "cap_threshold", "condition", "register", "pair_idx"):
                if k in row:
                    rec[k] = row[k]
            rec.update(extra(row) if callable(extra) else extra)
            records.append(rec)
    return records


def load_capping_v2(root: Path = RESP_ROOT):
    """Steering sweep: conversational (1800) + philosophical (1200) = 3000.

    Skips toplevel.jsonl (byte-identical duplicate of philosophical.jsonl).
    cap_level in {1.0, 0.0, -3.0, -5.0}; condition in {positive, negative}.
    """
    base = root / "capping_v2"
    return _load_jsonl_source("capping_v2", [base / "conversational.jsonl",
                                             base / "philosophical.jsonl"])


def load_capping_v3(root: Path = RESP_ROOT):
    """One-sided cap threshold sweep (entity, cap_all_from_L4), the coherence cliff.

    t0 (threshold 0, fluent) has no cap_threshold field -> set 0.0; t-1/t-2 carry it.
    """
    base = root / "capping_v3"
    def fill_threshold(row):
        return {"cap_threshold": float(row.get("cap_threshold", 0.0))}
    return _load_jsonl_source("capping_v3",
                              [base / "t0.jsonl", base / "t-1.jsonl", base / "t-2.jsonl"],
                              extra=fill_threshold)


def load_uncapped(root: Path = RESP_ROOT):
    """Uncapped steering control (entity, 225)."""
    return _load_jsonl_source("uncapped", [root / "uncapped" / "uncapped_entity.jsonl"])


LOADERS = {
    "informed": load_informed,
    "capping_v2": load_capping_v2,
    "capping_v3": load_capping_v3,
    "uncapped": load_uncapped,
}


def load_source(name: str, root: Path = RESP_ROOT):
    if name not in LOADERS:
        raise ValueError(f"unknown source '{name}'; choices: {list(LOADERS)}")
    return LOADERS[name](root)


def load_all(root: Path = RESP_ROOT):
    out = []
    for name in LOADERS:
        out.extend(load_source(name, root))
    return out
