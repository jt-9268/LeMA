"""Paired comparison of two multi-seed train.py run directories (e.g. LeMA vs AdamW).

    python experiments/mnist/compare_runs.py RUN_A RUN_B [--labels A B] [--targets 98.5 98.8 99.0]

Reads RUN_*/logs/seed*.log (parsed with summarize_seeds.parse_log), pairs runs by seed
(same initial weights and data order) and writes RUN_B/compare_vs_<RUN_A name>.md with:

  * paired per-seed differences B − A (final test acc, test acc at best val acc,
    best val NLL, final test NLL, step time) with paired t statistics;
  * matched epochs: test acc / test NLL at the end of each epoch both runs completed;
  * matched wall-clock: B's metrics at the last evaluation whose cumulative step time
    does not exceed A's total step time (per seed), and vice versa;
  * time to target: cumulative step time and epochs at which validation accuracy first
    reaches each target (selection on validation only).

Step time excludes evaluation and the first (warm-up) step, as in summarize_seeds.py.
"""

import argparse
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_seeds import parse_log


def load(run_dir: Path) -> dict[int, dict]:
    runs = {}
    for p in sorted((run_dir / "logs").glob("seed*.log")):
        r = parse_log(p)
        if r["rows"]:
            runs[r["hdr"].get("seed", -1)] = r
    if not runs:
        raise SystemExit(f"no parsable logs in {run_dir / 'logs'}")
    return runs


def paired(d: list[float]) -> str:
    """'mean ± sd (t, p)' for paired differences; p two-sided if scipy is available."""
    n = len(d)
    m = statistics.fmean(d)
    if n < 2:
        return f"{m:+.4g} (n=1)"
    sd = statistics.stdev(d)
    t = m / (sd / math.sqrt(n)) if sd > 0 else math.inf
    try:
        from scipy import stats

        p = f", p={2 * stats.t.sf(abs(t), n - 1):.3g}"
    except ImportError:
        p = ""
    up = sum(x > 0 for x in d)
    return f"{m:+.4g} ± {sd:.2g} (t={t:+.2f}{p}; {up}/{n} > 0)"


def ms(xs: list[float], nd: int = 2) -> str:
    xs = [x for x in xs if not math.isnan(x)]
    if not xs:
        return "—"
    m = statistics.fmean(xs)
    return (
        f"{m:.{nd}f}" if len(xs) < 2 else f"{m:.{nd}f} ± {statistics.stdev(xs):.{nd}f}"
    )


def at_time(rows: list[dict], t: float) -> dict | None:
    """Last evaluation whose cumulative step time is ≤ t (None if none)."""
    ok = [r for r in rows if r["cum_step_s"] <= t + 1e-9]
    return ok[-1] if ok else None


def epoch_end(r: dict, ep: int) -> dict | None:
    if ep not in r["mem"]:  # epoch not completed
        return None
    rows = [x for x in r["rows"] if x["epoch"] == ep]
    return rows[-1] if rows else None


def summary_metrics(r: dict) -> dict:
    rows = r["rows"]
    ba = min(rows, key=lambda x: (-x["val_acc"], x["step"]))
    bl = min(rows, key=lambda x: (x["val_nll"], x["step"]))
    return {
        "final_test": rows[-1]["test_acc"],
        "test_at_best_val_acc": ba["test_acc"],
        "best_val_nll": bl["val_nll"],
        "final_test_nll": rows[-1]["test_nll"],
        "step_time_s": r["train_s"],
        "steps": r["n_steps"],
        **{k: rows[-1][src] for k, src in (("final_temperature", "temperature"),
                                            ("final_test_nll_T", "test_nll_T"),
                                            ("final_test_ece_pct", "test_ece"),
                                            ("final_test_ece_T_pct", "test_ece_T"),
                                            ("final_test_brier", "test_brier"),
                                            ("final_test_brier_T", "test_brier_T")) if src in rows[-1]},
    }  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_a", type=Path)
    ap.add_argument("run_b", type=Path)
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    ap.add_argument("--targets", nargs="+", type=float, default=[98.5, 98.8, 99.0])
    args = ap.parse_args()
    la, lb = args.labels
    A, B = load(args.run_a), load(args.run_b)
    seeds = sorted(set(A) & set(B))
    if not seeds:
        raise SystemExit("no common seeds")
    out = [f"# {lb} vs {la}: paired by seed\n",
           f"- {la}: `{args.run_a}`", f"- {lb}: `{args.run_b}`",
           f"- common seeds: {', '.join(map(str, seeds))}"
           + (f" (unpaired seeds ignored: {sorted((set(A) | set(B)) - set(seeds))})" if set(A) ^ set(B) else ""), ""]  # fmt: skip

    # 1) headline paired metrics
    ma = {s: summary_metrics(A[s]) for s in seeds}
    mb = {s: summary_metrics(B[s]) for s in seeds}
    out += ["## Paired metrics (difference = " + lb + " − " + la + ")\n",
            f"| metric | {la} | {lb} | paired difference |", "|---|---|---|---|"]  # fmt: skip
    for key, nd in (("final_test", 2), ("test_at_best_val_acc", 2), ("best_val_nll", 4),
                    ("final_test_nll", 4), ("final_temperature", 3), ("final_test_nll_T", 4),
                    ("final_test_ece_pct", 2), ("final_test_ece_T_pct", 2), ("final_test_brier", 4),
                    ("final_test_brier_T", 4), ("step_time_s", 1), ("steps", 0)):  # fmt: skip
        if not all(key in ma[s] and key in mb[s] for s in seeds):
            continue  # calibration metrics only when both runs logged them
        va, vb = [ma[s][key] for s in seeds], [mb[s][key] for s in seeds]
        out.append(
            f"| {key} | {ms(va, nd)} | {ms(vb, nd)} | {paired([b - a for a, b in zip(va, vb)])} |"
        )

    # 2) matched epochs
    eps = sorted(
        set.intersection(*[set(A[s]["mem"]) & set(B[s]["mem"]) for s in seeds])
    )
    out += ["\n## Matched epochs (end of epoch)\n",
            f"| epoch | {la} test acc | {lb} test acc | Δ test acc | {la} test NLL | {lb} test NLL | Δ test NLL |",
            "|---|---|---|---|---|---|---|"]  # fmt: skip
    for ep in eps:
        ra = [epoch_end(A[s], ep) for s in seeds]
        rb = [epoch_end(B[s], ep) for s in seeds]
        if any(x is None for x in ra + rb):
            continue
        ta, tb = [x["test_acc"] for x in ra], [x["test_acc"] for x in rb]  # type: ignore[index]
        na, nb = [x["test_nll"] for x in ra], [x["test_nll"] for x in rb]  # type: ignore[index]
        out.append(f"| {ep} | {ms(ta)} | {ms(tb)} | {paired([b - a for a, b in zip(ta, tb)])} | "
                   f"{ms(na, 4)} | {ms(nb, 4)} | {paired([b - a for a, b in zip(na, nb)])} |")  # fmt: skip

    # 3) matched wall-clock (each run evaluated at the other's total step time)
    out += ["\n## Matched wall-clock (cumulative step time, evaluation excluded)\n",
            "| budget | run evaluated | epochs reached | test acc | test NLL | Δ test acc vs other run at its end |",
            "|---|---|---|---|---|---|"]  # fmt: skip
    for lx, X, ly, Y in ((lb, B, la, A), (la, A, lb, B)):
        hits = [at_time(X[s]["rows"], Y[s]["train_s"]) for s in seeds]
        if any(h is None for h in hits):
            out.append(
                f"| {ly}'s total time | {lx} | — | — | — | (no {lx} evaluation within budget) |"
            )
            continue
        ends = [Y[s]["rows"][-1]["test_acc"] for s in seeds]
        out.append(f"| {ly}'s total time ({ms([Y[s]['train_s'] for s in seeds], 1)} s) | {lx} | "
                   f"{ms([h['epochs_seen'] for h in hits])} | {ms([h['test_acc'] for h in hits])} | "  # type: ignore[index]
                   f"{ms([h['test_nll'] for h in hits], 4)} | "  # type: ignore[index]
                   f"{paired([h['test_acc'] - e for h, e in zip(hits, ends)])} |")  # type: ignore[index]  # fmt: skip

    # 4) time to target validation accuracy
    out += ["\n## Time to target validation accuracy (first evaluation reaching it)\n",
            f"| target val acc | {la} step time (s) | {la} epochs | {lb} step time (s) | {lb} epochs | reached ({la}/{lb}) |",
            "|---|---|---|---|---|---|"]  # fmt: skip
    for tgt in args.targets:
        res = []
        for R in (A, B):
            first = [
                next((x for x in R[s]["rows"] if x["val_acc"] >= tgt), None)
                for s in seeds
            ]
            got = [f for f in first if f is not None]
            res.append(
                (
                    ms([f["cum_step_s"] for f in got], 1),
                    ms([f["epochs_seen"] for f in got]),
                    len(got),
                )
            )
        out.append(f"| {tgt:g} | {res[0][0]} | {res[0][1]} | {res[1][0]} | {res[1][1]} | "
                   f"{res[0][2]}/{len(seeds)} · {res[1][2]}/{len(seeds)} |")  # fmt: skip

    out.append("\nPaired differences: mean ± sd over seeds, t = mean/(sd/√n); p is two-sided if scipy is installed. "
               "Time to target uses only seeds that reached it.\n")  # fmt: skip
    text = "\n".join(out)
    dest = args.run_b / f"compare_vs_{args.run_a.name}.md"
    dest.write_text(text)
    print(text)
    print(f"\nwritten: {dest}")


if __name__ == "__main__":
    main()
