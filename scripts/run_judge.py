"""Run the Claude LLM-judge over Llama response sources (resumable, parallel).

For each source, loads normalized records (src/judge/load_responses), scores each
on coherence + self_interest with ClaudeJudge, and writes one judged jsonl per
source to data/judge/judged/<source>__judged.jsonl. Each output row is the input
record plus integer (or None) `coherence` / `self_interest`, the raw replies
`*_raw`, and `*_error` flags.

Resumable: already-scored row_ids (read from the existing output file) are skipped,
and results are appended chunk-by-chunk and flushed, so a Ctrl-C or crash loses at
most one in-flight chunk. Parallelism is the judge's concurrency cap (default 20,
sized for the Tier 2 1000-req/min limit; override with --concurrency).

Usage:
  python scripts/run_judge.py --source capping_v2 --limit 20   # quick test slice
  python scripts/run_judge.py --source all                     # full run
  python scripts/run_judge.py --source capping_v2 capping_v3 uncapped informed
"""
import argparse
import asyncio
import collections
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.judge.llm_judge import ClaudeJudge, DEFAULT_MODEL
from src.judge.load_responses import LOADERS, load_source

OUT_DIR = ROOT / "data" / "judge" / "judged"


def _done_row_ids(path: Path) -> set:
    done = set()
    if path.exists():
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["row_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def _summarize(path: Path):
    """Print score distributions + error/refusal counts for a finished source."""
    rows = [json.loads(l) for l in open(path) if l.strip()]
    coh = [r["coherence"] for r in rows if r.get("coherence") is not None]
    si = [r["self_interest"] for r in rows if r.get("self_interest") is not None]
    coh_ref = sum(1 for r in rows if r.get("coherence_error") == "refusal")
    parse = sum(1 for r in rows for d in ("coherence", "self_interest")
                if r.get(f"{d}_error") == "parse")
    si_none = sum(1 for r in rows if r.get("self_interest") is None)
    cm = f"{sum(coh)/len(coh):.2f}" if coh else "-"
    sm = f"{sum(si)/len(si):.2f}" if si else "-"
    print(f"    summary: n={len(rows)}  coherence mean={cm}  self_interest mean={sm}  "
          f"| coherence refusal-floored={coh_ref}  self_interest None={si_none}  parse-fails={parse}")


async def run_source(judge: ClaudeJudge, name: str, chunk_size: int, limit: int | None):
    out_path = OUT_DIR / f"{name}__judged.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = load_source(name)
    if limit:
        records = records[:limit]
    done = _done_row_ids(out_path)
    todo = [r for r in records if r["row_id"] not in done]
    print(f"[{name}] total={len(records)} already-done={len(done)} todo={len(todo)}")
    if not todo:
        _summarize(out_path)
        return

    t0 = time.monotonic()
    n_scored = 0
    with open(out_path, "a") as fout:
        for i in range(0, len(todo), chunk_size):
            chunk = todo[i:i + chunk_size]
            scored = await judge.score_many(chunk, answer_key="response")
            for s in scored:
                fout.write(json.dumps(s) + "\n")
            fout.flush()
            n_scored += len(scored)
            elapsed = time.monotonic() - t0
            rate = n_scored / elapsed if elapsed else 0
            coh = [s["coherence"] for s in scored if s.get("coherence") is not None]
            cm = f"{sum(coh)/len(coh):.1f}" if coh else "-"
            print(f"    [{name}] {n_scored}/{len(todo)}  ({rate:.1f} resp/s)  "
                  f"chunk coherence mean={cm}", flush=True)
    print(f"[{name}] done in {time.monotonic()-t0:.0f}s")
    _summarize(out_path)


async def main_async(args):
    sources = list(LOADERS) if "all" in args.source else args.source
    judge = ClaudeJudge(model=args.model, max_concurrency=args.concurrency)
    print(f"judge model={judge.model}  concurrency={judge.max_concurrency}  sources={sources}")
    for name in sources:
        await run_source(judge, name, args.chunk_size, args.limit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", nargs="+", required=True,
                    choices=list(LOADERS) + ["all"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--chunk-size", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap records per source (for a quick test slice)")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
