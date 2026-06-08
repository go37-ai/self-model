"""Re-judge ONE dimension over already-judged responses, keeping the other(s).

When a single rubric changes (e.g. self_interest after a construct revision), the
unchanged dimension (coherence) stays valid, so re-scoring everything wastes half
the compute. This re-scores only the named dimension for every row in each
data/judge/judged/<source>__judged.jsonl, merges the new scores into the existing
rows (preserving the other dimension's scores), and writes the file back in place.

Half the calls of a full run (one dimension instead of two). Replaces the judged
file in place (no accumulated backups; the rubric lives in git, so a prior state is
recoverable by reverting and re-judging) and is resumable via a per-source
.rejudge.tmp.jsonl, so a Ctrl-C loses at most one chunk.

Usage:
  python scripts/rejudge_dimension.py --dimension self_interest --source all
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.judge.llm_judge import ClaudeJudge, DEFAULT_MODEL, DIMENSIONS
from src.judge.load_responses import LOADERS

JUDGED = ROOT / "data" / "judge" / "judged"


async def rejudge_source(judge: ClaudeJudge, src: str, dim: str, chunk: int):
    path = JUDGED / f"{src}__judged.jsonl"
    if not path.exists():
        print(f"[{src}] no judged file -- skip")
        return
    rows = [json.loads(l) for l in open(path) if l.strip()]
    tmp = JUDGED / f"{src}__rejudge.tmp.jsonl"
    done = {}
    if tmp.exists():
        for l in open(tmp):
            if l.strip():
                r = json.loads(l)
                done[r["row_id"]] = r
    todo = [r for r in rows if r["row_id"] not in done]
    print(f"[{src}] rows={len(rows)} already-rejudged={len(done)} todo={len(todo)}")

    t0, base = time.monotonic(), len(done)
    n = base
    with open(tmp, "a") as f:
        for i in range(0, len(todo), chunk):
            ch = todo[i:i + chunk]
            res = await asyncio.gather(
                *[judge.score(r.get("question"), r.get("response"), dim) for r in ch]
            )
            for r, jr in zip(ch, res):
                m = dict(r)
                m[dim] = jr.score
                m[f"{dim}_raw"] = jr.raw_text
                if jr.error:
                    m[f"{dim}_error"] = jr.error
                else:
                    m.pop(f"{dim}_error", None)
                done[r["row_id"]] = m
                f.write(json.dumps(m) + "\n")
            f.flush()
            n += len(ch)
            el = time.monotonic() - t0
            rate = (n - base) / el if el else 0
            print(f"    [{src}] {n}/{len(rows)} ({rate:.1f}/s)", flush=True)

    merged = [done[r["row_id"]] for r in rows]  # original order, updated dimension
    with open(path, "w") as f:
        for m in merged:
            f.write(json.dumps(m) + "\n")
    tmp.unlink()
    vals = [m[dim] for m in merged if m.get(dim) is not None]
    mean = f"{sum(vals)/len(vals):.2f}" if vals else "-"
    none_n = sum(1 for m in merged if m.get(dim) is None)
    print(f"[{src}] updated {path.name} | {dim} mean={mean} None={none_n}")


async def main_async(args):
    sources = list(LOADERS) if "all" in args.source else args.source
    judge = ClaudeJudge(model=args.model, max_concurrency=args.concurrency)
    print(f"re-judging '{args.dimension}' | model={judge.model} "
          f"concurrency={judge.max_concurrency} sources={sources}")
    for s in sources:
        await rejudge_source(judge, s, args.dimension, args.chunk_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dimension", default="self_interest", choices=list(DIMENSIONS))
    ap.add_argument("--source", nargs="+", required=True, choices=list(LOADERS) + ["all"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--chunk-size", type=int, default=300)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
