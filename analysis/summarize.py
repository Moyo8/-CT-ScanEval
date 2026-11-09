"""Summarise all runs in results/ into results/summary.md (+ a chart).

    python analysis/summarize.py              # every run
    python analysis/summarize.py --latest     # one run per model: the most complete, newest if tied

Metrics (per model):
    pass rate   share of all trials that passed
    mean score  mean partial-credit score over all trials
    pass@k      share of tasks passed in at least one of k trials
    pass^k      share of tasks passed in every one of k trials (consistency)
    spread      min-max pass rate across individual trial rounds
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_runs(latest):
    runs = []
    for d in sorted((ROOT / "results").iterdir()):
        if not (d / "run.json").exists():
            continue
        meta = json.loads((d / "run.json").read_text())
        recs = [json.loads(line) for line in (d / "results.jsonl").read_text().splitlines() if line]
        effort = meta["settings"].get("effort") or meta["settings"].get("reasoning_effort") or ""
        meta["label"] = f"{meta['provider']}:{meta['model']}" + (
            f" (effort={effort})" if effort and "default" not in effort else "")
        runs.append((meta, recs))
    if latest:
        # one run per model: the most complete one (most trials), newest if tied
        keep = {}
        for meta, recs in runs:  # sorted by run id = time
            best = keep.get(meta["label"])
            if best is None or len(recs) >= len(best[1]):
                keep[meta["label"]] = (meta, recs)
        runs = list(keep.values())
    return runs


def stats(recs):
    by_task = defaultdict(list)
    by_trial = defaultdict(list)
    for r in recs:
        by_task[r["task_id"]].append(r)
        by_trial[r["trial"]].append(r["passed"])
    k = max(len(v) for v in by_task.values())
    k_min = min(len(v) for v in by_task.values())
    return dict(
        k=k,
        k_label=str(k) if k_min == k else f"{k_min}-{k}",
        n=len(recs),
        pass_rate=sum(r["passed"] for r in recs) / len(recs),
        mean_score=sum(r["score"] for r in recs) / len(recs),
        pass_at_k=sum(any(r["passed"] for r in v) for v in by_task.values()) / len(by_task),
        pass_hat_k=sum(all(r["passed"] for r in v) for v in by_task.values()) / len(by_task),
        spread=(min(sum(v) / len(v) for v in by_trial.values()), max(sum(v) / len(v) for v in by_trial.values())),
        by_task={t: (sum(r["passed"] for r in v), len(v), sum(r["score"] for r in v) / len(v)) for t, v in by_task.items()},
        errors=sum(1 for r in recs if r["error"]),
    )


def pct(x):
    return f"{100 * x:.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latest", action="store_true")
    args = ap.parse_args()
    runs = load_runs(args.latest)
    if not runs:
        raise SystemExit("no runs in results/")
    tasks = sorted({r["task_id"] for _, recs in runs for r in recs})

    lines = ["# CT-ScanEval results", "",
             "| Model (version, settings) | Served model | Trials/task | Pass rate (spread) | Mean score | pass@k | pass^k | Cost per full run | Runtime |",
             "|---|---|---|---|---|---|---|---|---|"]
    table = []
    for meta, recs in runs:
        s = stats(recs)
        table.append((meta, s))
        cost = f"${meta['cost_usd']:.2f}" if meta.get("cost_usd") is not None else "n/a"
        lines.append(
            f"| {meta['label']} | {', '.join(meta['served_models'])} | {s['k_label']} | {pct(s['pass_rate'])} "
            f"({pct(s['spread'][0])}-{pct(s['spread'][1])}) | {s['mean_score']:.2f} | {pct(s['pass_at_k'])} | "
            f"{pct(s['pass_hat_k'])} | {cost} | {meta['runtime_s']:.0f}s |")
    lines += ["", "## Per task (passed trials / trials, mean partial-credit score)", "",
              "| Task | " + " | ".join(m["label"] for m, _ in table) + " |",
              "|---|" + "---|" * len(table)]
    for t in tasks:
        cells = []
        for _, s in table:
            p, n, sc = s["by_task"].get(t, (0, 0, 0))
            cells.append(f"{p}/{n} ({sc:.2f})" if n else "-")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    lines += ["", "Run ids: " + ", ".join(f"`{m['run_id']}`" for m, _ in table), ""]
    (ROOT / "results" / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))

    # chart: per-task mean score by model
    charts = ROOT / "results" / "charts"
    charts.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    width = 0.8 / len(table)
    for i, (meta, s) in enumerate(table):
        ax.bar([x + i * width for x in range(len(tasks))],
               [s["by_task"].get(t, (0, 0, 0))[2] for t in tasks], width, label=meta["label"])
    ax.set_xticks([x + width * (len(table) - 1) / 2 for x in range(len(tasks))])
    ax.set_xticklabels([t.split("_", 1)[1].replace("_", "\n") for t in tasks], fontsize=8)
    ax.set_ylabel("mean score (0-1)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("CT-ScanEval: mean partial-credit score per task")
    fig.tight_layout()
    fig.savefig(charts / "per_task_scores.png", dpi=120)


if __name__ == "__main__":
    main()
