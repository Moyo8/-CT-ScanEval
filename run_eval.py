"""CT-ScanEval harness: run every task k times against one model, grade each
trial, and save transcripts + results.

Examples
    python run_eval.py --provider oracle                     # no API key needed
    python run_eval.py --provider random --trials 5           # chance baseline
    python run_eval.py --provider anthropic --model claude-opus-5-5 --effort high --trials 3
    python run_eval.py --provider openai --model <openai-vision-model> --trials 3
    python run_eval.py --provider gemini --model gemini-3.8-flash --trials 3 --pace 7   # free tier

Outputs
    results/<run_id>/results.jsonl   one line per trial (grade + usage)
    results/<run_id>/run.json        run metadata (model, settings, dates, cost)
    transcripts/<run_id>/<task>__trial<k>.json   full prompt + raw reply
Then:  python analysis/summarize.py
"""
import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from graders.grader import grade  # noqa: E402
from harness.providers import PROVIDERS, QuotaExhausted  # noqa: E402


def load_env(path=ROOT / ".env"):
    """Minimal .env loader (KEY=value lines) so no extra dependency is needed."""
    import os

    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_tasks(selected=None):
    tasks = []
    for d in sorted((ROOT / "tasks").iterdir()):
        if (d / "task.json").exists():
            t = json.loads((d / "task.json").read_text())
            if not selected or t["id"] in selected:
                t["_dir"] = d
                tasks.append(t)
    return tasks


def price_for(model):
    prices = json.loads((ROOT / "harness" / "pricing.json").read_text())["usd_per_million_tokens"]
    return prices.get(model)


def run_trial(provider, task, trial, run_id):
    images = [task["_dir"] / p for p in task["images"]]
    t0 = time.time()
    error = None
    try:
        out = provider(task, images, task["image_labels"], task["prompt"])
    except QuotaExhausted:
        raise
    except Exception as exc:  # recorded, not hidden: an API error is a failed trial
        out = dict(text="", served_model=None, stop_reason="api_error",
                   usage=dict(input_tokens=None, output_tokens=None), request_id=None)
        error = f"{type(exc).__name__}: {exc}"
    latency = round(time.time() - t0, 2)
    g = grade(task["id"], out["text"])
    if error:
        g["error"] = error
    elif out["stop_reason"] == "refusal":
        g["error"] = "model refused"
    record = dict(
        run_id=run_id,
        task_id=task["id"],
        trial=trial,
        provider=provider.name,
        model=provider.model,
        served_model=out["served_model"],
        stop_reason=out["stop_reason"],
        passed=g["passed"],
        score=g["score"],
        checks=g["checks"],
        details=g["details"],
        error=g["error"],
        usage=out["usage"],
        latency_s=latency,
        timestamp=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    )
    transcript = dict(
        record,
        prompt=task["prompt"],
        images=[dict(label=lab, file=str(p.relative_to(ROOT)),
                     sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for lab, p in zip(task["image_labels"], images)],
        response_text=out["text"],
        request_id=out["request_id"],
    )
    return record, transcript


def main():
    ap = argparse.ArgumentParser(description="Run CT-ScanEval against one model.")
    ap.add_argument("--provider", required=True, choices=sorted(PROVIDERS))
    ap.add_argument("--model", help="model ID as the provider's API names it")
    ap.add_argument("--effort", help="reasoning effort (anthropic: low..max; openai: reasoning_effort)")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--tasks", nargs="*", help="task ids (default: all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pace", type=float, default=0,
                    help="seconds between calls, one at a time (use ~7 for free-tier rate limits)")
    ap.add_argument("--tag", default="", help="optional label added to the run id")
    args = ap.parse_args()

    load_env()
    if args.provider in ("anthropic", "openai", "gemini") and not args.model:
        ap.error("--model is required for API providers")
    model = args.model or ("reference" if args.provider == "oracle" else "seed0")
    provider = PROVIDERS[args.provider](model, args.effort)
    tasks = load_tasks(args.tasks)

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = model.replace("/", "_")
    run_id = f"{stamp}__{args.provider}__{safe_model}" + (f"__{args.effort}" if args.effort else "") + (f"__{args.tag}" if args.tag else "")
    res_dir = ROOT / "results" / run_id
    tr_dir = ROOT / "transcripts" / run_id
    res_dir.mkdir(parents=True)
    tr_dir.mkdir(parents=True)

    # trial-major order: if a daily quota runs out, the finished trials are complete rounds
    jobs = [(t, k) for k in range(1, args.trials + 1) for t in tasks]
    print(f"{run_id}: {len(tasks)} tasks x {args.trials} trials = {len(jobs)} calls")
    started = time.time()
    records = []
    incomplete = None

    def save(rec, tr):
        records.append(rec)
        (tr_dir / f"{rec['task_id']}__trial{rec['trial']}.json").write_text(json.dumps(tr, indent=2) + "\n")
        mark = "PASS" if rec["passed"] else "fail"
        print(f"  {rec['task_id']:<32} trial {rec['trial']}  {mark}  score={rec['score']:.2f}"
              + (f"  [{rec['error']}]" if rec["error"] else ""), flush=True)

    if args.pace > 0 or args.workers == 1:
        for i, (t, k) in enumerate(jobs):
            t0 = time.time()
            try:
                save(*run_trial(provider, t, k, run_id))
            except QuotaExhausted as exc:
                incomplete = f"daily quota reached after {len(records)} of {len(jobs)} calls: {exc}"
                break
            if args.pace and i < len(jobs) - 1:
                time.sleep(max(0.0, args.pace - (time.time() - t0)))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_trial, provider, t, k, run_id) for t, k in jobs]
            for fut in as_completed(futures):
                try:
                    save(*fut.result())
                except QuotaExhausted as exc:
                    incomplete = f"daily quota reached after {len(records)} of {len(jobs)} calls: {exc}"
                    for f in futures:
                        f.cancel()
    if not records:
        raise SystemExit(f"no trials completed. {incomplete or ''}")

    records.sort(key=lambda r: (r["task_id"], r["trial"]))
    with open(res_dir / "results.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    tok_in = sum(r["usage"]["input_tokens"] or 0 for r in records)
    tok_out = sum(r["usage"]["output_tokens"] or 0 for r in records)
    price = price_for(model)
    cost = round(tok_in / 1e6 * price["input"] + tok_out / 1e6 * price["output"], 4) if price else None
    served = sorted({r["served_model"] for r in records if r["served_model"]})
    meta = dict(
        run_id=run_id,
        provider=args.provider,
        model=model,
        served_models=served,
        settings=provider.settings(),
        trials_per_task=args.trials,
        tasks=[t["id"] for t in tasks],
        started_utc=stamp,
        runtime_s=round(time.time() - started, 1),
        input_tokens=tok_in,
        output_tokens=tok_out,
        cost_usd=cost,
        cost_note=None if price else "no price for this model in harness/pricing.json",
        api_errors=sum(1 for r in records if r["stop_reason"] == "api_error"),
        trials_completed=len(records),
        incomplete=incomplete,
    )
    (res_dir / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    n = len(records)
    print(f"pass rate {sum(r['passed'] for r in records)}/{n}; mean score {sum(r['score'] for r in records)/n:.2f}; "
          f"tokens in/out {tok_in}/{tok_out}; cost {cost if cost is not None else 'n/a'} USD")
    if incomplete:
        print(f"WARNING: run stopped early - {incomplete[:200]}")
    if meta["api_errors"]:
        print(f"WARNING: {meta['api_errors']} API errors - see transcripts before reporting results")


if __name__ == "__main__":
    main()
