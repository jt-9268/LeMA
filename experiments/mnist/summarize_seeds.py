"""Summarise several train.py runs (one log per seed) into per-step and per-seed tables.

    python experiments/mnist/summarize_seeds.py RUN_DIR

Reads RUN_DIR/logs/seed*.log (stdout of train.py, captured with tee) and writes:

    RUN_DIR/steps.csv    one row per (seed, LM step): losses, accuracies, LM stats, step time
    RUN_DIR/summary.md   per-seed selection table, per-epoch and per-step mean ± std over seeds
    RUN_DIR/steps.png    mean ± std curves vs epochs (only if matplotlib is importable)

Step time is the gap between a step's log line and the previous log line (an eval,
epoch or header line), so it excludes evaluation. Step 1 includes warm-up and is
excluded from medians. Statistics use the sample std (n − 1).
"""

import csv
import datetime as dt
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

TS = r"(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)"
NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
RE_STEP = re.compile(
    TS
    + r" \| step info:? \|? ?epoch: (?P<epoch>\d+) \| batch: \[(?P<b0>\d+), (?P<b1>\d+)\] \| "
    rf"loss: (?P<loss>{NUM}) \| method: \w+ \| iterations: (?P<it>\d+) \| (?:damp|lr): (?P<damp>{NUM})"
)
RE_EVAL = re.compile(
    TS + r" \| eval info \| epoch: (?P<epoch>\d+) \| step: (?P<step>\d+) \| "
    rf"val loss: (?P<vl>{NUM}) \| val accuracy: (?P<va>{NUM})% \| "
    rf"test loss: (?P<tl>{NUM}) \| test accuracy: (?P<ta>{NUM})%"
)
RE_EPOCH = re.compile(
    TS
    + rf" \| epoch info \| epoch: (?P<epoch>\d+) \| loss: {NUM} \| memory: (?P<mem>{NUM}) GB"
)
RE_CALIB = re.compile(
    TS + r" \| calib info \| epoch: \d+ \| step: (?P<step>\d+) \| "
    rf"temperature: (?P<T>{NUM}) \| val ece: (?P<vece>{NUM})% \| val ece@T: (?P<vecet>{NUM})% \| "
    rf"test nll: (?P<tnll>{NUM}) \| test nll@T: (?P<tnllt>{NUM}) \| "
    rf"test ece: (?P<tece>{NUM})% \| test ece@T: (?P<tecet>{NUM})% \| "
    rf"test brier: (?P<tb>{NUM}) \| test brier@T: (?P<tbt>{NUM}) \| "
    rf"test conf: (?P<tc>{NUM})% \| test conf@T: (?P<tct>{NUM})%"
)
CALIB_KEYS = {"T": "temperature", "vece": "val_ece", "vecet": "val_ece_T", "tnll": "test_nll_c",
              "tnllt": "test_nll_T", "tece": "test_ece", "tecet": "test_ece_T", "tb": "test_brier",
              "tbt": "test_brier_T", "tc": "test_conf", "tct": "test_conf_T"}  # fmt: skip
RE_ANY = re.compile(TS + r" \| ")
RE_HDR = {
    "batch_size": re.compile(r"Batch size: ([\d,]+)"),
    "seed": re.compile(r"seed: (\d+)"),
    "activation": re.compile(r"Activation: (\w+)"),
    "init": re.compile(r"init: (\w+)"),
    "damp_start": re.compile(r"damp start: ([0-9.eE+-]+)"),
    "damp_min": re.compile(r"damp min: ([0-9.eE+-]+)"),
    "label_smoothing": re.compile(r"label smoothing: ([0-9.eE+-]+)"),
    "optimizer": re.compile(r"Optimizer: (\w+)"),
    "lr": re.compile(r"Optimizer: \w+ \| lr: ([0-9.eE+-]+)"),
    "weight_decay": re.compile(r"weight decay: ([0-9.eE+-]+)"),
    "train_size": re.compile(r"Train / val / test samples: ([\d,]+)"),
}


def _t(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")  # noqa: DTZ007 — only differences are used


def parse_log(path: Path) -> dict:
    """Return header fields, per-step rows and per-epoch peak memory for one log."""
    hdr, rows, mem = {}, [], {}
    last_ts, pending = None, None
    n_step_lines, cum_step_s, first_step_s = 0, 0.0, float("nan")
    for line in path.read_text(errors="replace").splitlines():
        for key, rx in RE_HDR.items():
            if key not in hdr and (m := rx.search(line)):
                hdr[key] = (
                    int(m.group(1).replace(",", ""))
                    if key in ("batch_size", "seed", "train_size")
                    else m.group(1)
                )
        if m := RE_STEP.search(line):
            ts = _t(m["ts"])
            gap = (ts - last_ts).total_seconds() if last_ts else float("nan")
            n_step_lines += 1
            if n_step_lines == 1:
                first_step_s = gap  # includes warm-up / compilation
            else:
                cum_step_s += gap
            pending = {
                "epoch": int(m["epoch"]), "samples_in_epoch": int(m["b1"]), "batch": int(m["b1"]) - int(m["b0"]) + 1,
                "train_batch_nll": float(m["loss"]), "lm_iterations": int(m["it"]), "damp": float(m["damp"]),
                "step_s": gap, "cum_step_s": cum_step_s,
            }  # fmt: skip
            last_ts = ts
        elif m := RE_EVAL.search(line):
            last_ts = _t(m["ts"])
            if pending is None:
                continue
            pending.update(step=int(m["step"]), val_nll=float(m["vl"]), val_acc=float(m["va"]),
                           test_nll=float(m["tl"]), test_acc=float(m["ta"]))  # fmt: skip
            rows.append(pending)
            pending = None
        elif m := RE_CALIB.search(line):
            last_ts = _t(m["ts"])
            if rows and rows[-1]["step"] == int(m["step"]):
                rows[-1].update({v: float(m[k]) for k, v in CALIB_KEYS.items()})
        elif m := RE_EPOCH.search(line):
            last_ts = _t(m["ts"])
            mem[int(m["epoch"])] = float(m["mem"])
        elif m := RE_ANY.match(line):
            last_ts = _t(m["ts"])
    n_train = hdr.get("train_size")
    for r in rows:
        r["epochs_seen"] = (
            (r["epoch"] - 1) + r["samples_in_epoch"] / n_train
            if n_train
            else float("nan")
        )
    return {"hdr": hdr, "rows": rows, "mem": mem, "train_s": cum_step_s,
            "first_step_s": first_step_s, "n_steps": n_step_lines}  # fmt: skip


def ms(xs: list[float], nd: int = 2) -> str:
    xs = [x for x in xs if not math.isnan(x)]
    if not xs:
        return "—"
    m = statistics.fmean(xs)
    return (
        f"{m:.{nd}f}" if len(xs) < 2 else f"{m:.{nd}f} ± {statistics.stdev(xs):.{nd}f}"
    )


def summarize(run_dir: Path) -> str:
    logs = sorted(
        (run_dir / "logs").glob("seed*.log"),
        key=lambda p: int(re.sub(r"\D", "", p.stem) or 0),
    )
    if not logs:
        raise SystemExit(f"no logs in {run_dir / 'logs'}")
    runs = {}
    for p in logs:
        r = parse_log(p)
        seed = r["hdr"].get("seed", int(re.sub(r"\D", "", p.stem)))
        if r["rows"]:
            runs[seed] = r

    # steps.csv
    fields = ["seed", "step", "epoch", "epochs_seen", "batch", "train_batch_nll", "val_nll", "val_acc",
              "test_nll", "test_acc", "lm_iterations", "damp", "step_s", "cum_step_s",
              *CALIB_KEYS.values()]  # fmt: skip
    with open(run_dir / "steps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for seed, r in runs.items():
            for row in r["rows"]:
                w.writerow({"seed": seed, **row})

    # per-seed selection table
    per_seed, out = [], []
    for seed, r in runs.items():
        rows = r["rows"]
        bl = min(rows, key=lambda x: (x["val_nll"], x["step"]))
        ba = min(rows, key=lambda x: (-x["val_acc"], x["step"]))
        steady = [x["step_s"] for x in rows[1:] if x["batch"] == rows[0]["batch"]]
        per_seed.append({
            "seed": seed, "epochs": len(r["mem"]), "steps": len(rows),  # completed epochs
            "best_val_nll": bl["val_nll"], "best_val_nll_step": bl["step"], "test_at_best_nll": bl["test_acc"],
            "best_val_acc": ba["val_acc"], "best_val_acc_step": ba["step"], "test_at_best_acc": ba["test_acc"],
            "final_test": rows[-1]["test_acc"], "final_val": rows[-1]["val_acc"],
            "step_s": statistics.median(steady) if steady else float("nan"),
            "lm_it": statistics.fmean(x["lm_iterations"] for x in rows),
            "train_s": r["train_s"], "n_steps": r["n_steps"],
            "mem": max(r["mem"].values()) if r["mem"] else float("nan"),
        })  # fmt: skip
    h0 = runs[next(iter(runs))]["hdr"]
    out.append(f"# train.py multi-seed summary: {run_dir.name}\n")
    out.append(f"{len(runs)} seed(s): {', '.join(map(str, runs))} · global batch {h0.get('batch_size', '?'):,} · "
               f"activation {h0.get('activation', '?')} · init {h0.get('init', '?')} · " + (
                   f"optimizer adamw · peak lr {h0.get('lr', '?')} · weight decay {h0.get('weight_decay', '?')}"
                   if h0.get("optimizer") == "adamw"
                   else f"optimizer lema · damp start {h0.get('damp_start', '0.001 (not logged)')} · "
                        f"damp min {h0.get('damp_min', '0.1 (not logged)')}")
               + f" · label smoothing {h0.get('label_smoothing', '0 (not logged)')}\n")  # fmt: skip
    out.append("## Per seed\n")
    out.append("| seed | epochs done | best val NLL (step) | test acc @ best val NLL | best val acc (step) | "
               "test acc @ best val acc | final test acc | steps | step time total (s) | median step (s) | "
               "LM it/step | peak mem (GB) |")  # fmt: skip
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in per_seed:
        out.append(f"| {s['seed']} | {s['epochs']} | {s['best_val_nll']:.4f} ({s['best_val_nll_step']}) | "
                   f"{s['test_at_best_nll']:.2f} | {s['best_val_acc']:.2f} ({s['best_val_acc_step']}) | "
                   f"{s['test_at_best_acc']:.2f} | {s['final_test']:.2f} | {s['n_steps']} | {s['train_s']:.1f} | "
                   f"{s['step_s']:.3f} | {s['lm_it']:.2f} | "
                   f"{s['mem']:.2f} |")  # fmt: skip
    col = lambda k: [s[k] for s in per_seed]
    out.append(f"| **mean ± std** | | {ms(col('best_val_nll'), 4)} ({ms(col('best_val_nll_step'), 1)}) | "
               f"{ms(col('test_at_best_nll'))} | {ms(col('best_val_acc'))} ({ms(col('best_val_acc_step'), 1)}) | "
               f"{ms(col('test_at_best_acc'))} | {ms(col('final_test'))} | {ms(col('n_steps'), 0)} | {ms(col('train_s'), 1)} | "
               f"{ms(col('step_s'), 3)} | "
               f"{ms(col('lm_it'))} | {ms(col('mem'))} |\n")  # fmt: skip

    # per-epoch-end and per-step tables (aligned on step index; same B ⇒ same steps per epoch)
    by_step = defaultdict(list)
    for r in runs.values():
        for row in r["rows"]:
            by_step[row["step"]].append(row)
    # epoch ends: last step of each *completed* epoch (an "epoch info" line was logged), per seed
    ends = defaultdict(list)
    for r in runs.values():
        for ep in r["mem"]:
            in_ep = [x for x in r["rows"] if x["epoch"] == ep]
            if in_ep:
                ends[ep].append(in_ep[-1])
    out.append("## End of each completed epoch (mean ± std over seeds)\n")
    out.append(
        "| epoch | n seeds | step | val acc % | test acc % | val NLL | test NLL | train batch NLL |"
    )
    out.append("|---|---|---|---|---|---|---|---|")
    for ep in sorted(ends):
        rs = ends[ep]
        out.append(f"| {ep} | {len(rs)} | {ms([x['step'] for x in rs], 0)} | {ms([x['val_acc'] for x in rs])} | "
                   f"{ms([x['test_acc'] for x in rs])} | {ms([x['val_nll'] for x in rs], 4)} | "
                   f"{ms([x['test_nll'] for x in rs], 4)} | {ms([x['train_batch_nll'] for x in rs], 5)} |")  # fmt: skip
    if any("temperature" in x for r in runs.values() for x in r["rows"]):
        out.append(
            "\n## Calibration (temperature T fitted on validation, applied to test)\n"
        )
        out.append(
            "End of each completed epoch, mean ± std over seeds. ECE: 15 equal-width bins.\n"
        )
        out.append("| epoch | n | T | test NLL | test NLL @T | test ECE % | test ECE @T % | test Brier | "
                   "test Brier @T | test conf % | test conf @T % |")  # fmt: skip
        out.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for ep in sorted(ends):
            rs = [x for x in ends[ep] if "temperature" in x]
            if not rs:
                continue
            c = lambda k, nd=2, rs=rs: ms([x[k] for x in rs], nd)
            out.append(f"| {ep} | {len(rs)} | {c('temperature', 3)} | {c('test_nll_c', 4)} | {c('test_nll_T', 4)} | "
                       f"{c('test_ece')} | {c('test_ece_T')} | {c('test_brier', 4)} | {c('test_brier_T', 4)} | "
                       f"{c('test_conf')} | {c('test_conf_T')} |")  # fmt: skip
        out.append("\nPer seed, final evaluation:\n")
        out.append("| seed | T | test NLL → @T | test ECE % → @T | test Brier → @T |")
        out.append("|---|---|---|---|---|")
        for seed, r in runs.items():
            x = r["rows"][-1]
            if "temperature" in x:
                out.append(f"| {seed} | {x['temperature']:.3f} | {x['test_nll_c']:.4f} → {x['test_nll_T']:.4f} | "
                           f"{x['test_ece']:.2f} → {x['test_ece_T']:.2f} | "
                           f"{x['test_brier']:.4f} → {x['test_brier_T']:.4f} |")  # fmt: skip
    out.append("\n## Every step (mean ± std over seeds)\n")
    out.append(
        "| step | epochs seen | n | val acc % | test acc % | val NLL | test NLL | step s |"
    )
    out.append("|---|---|---|---|---|---|---|---|")
    for st in sorted(by_step):
        rs = by_step[st]
        out.append(f"| {st} | {ms([x['epochs_seen'] for x in rs])} | {len(rs)} | {ms([x['val_acc'] for x in rs])} | "
                   f"{ms([x['test_acc'] for x in rs])} | {ms([x['val_nll'] for x in rs], 4)} | "
                   f"{ms([x['test_nll'] for x in rs], 4)} | {ms([x['step_s'] for x in rs], 3)} |")  # fmt: skip
    out.append("\nStep time total = sum of all step times except step 1 (warm-up), evaluation excluded. "
               "Selection uses validation only; step 1 times include warm-up. "
               "train batch NLL is measured on the batch just fitted (optimistic).\n")  # fmt: skip
    text = "\n".join(out)
    (run_dir / "summary.md").write_text(text)

    try:  # optional figure
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return text
    steps = sorted(by_step)
    x = [statistics.fmean(r["epochs_seen"] for r in by_step[s]) for s in steps]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for key, color, label in (
        ("val_acc", "#888780", "val"),
        ("test_acc", "#2a78d6", "test"),
    ):
        m = [statistics.fmean(r[key] for r in by_step[s]) for s in steps]
        sd = [
            statistics.stdev([r[key] for r in by_step[s]]) if len(by_step[s]) > 1 else 0
            for s in steps
        ]
        ax[0].plot(x, m, marker="o", ms=3, color=color, label=f"{label} acc")
        ax[0].fill_between(
            x,
            [a - b for a, b in zip(m, sd)],
            [a + b for a, b in zip(m, sd)],
            color=color,
            alpha=0.2,
        )
    for key, color, label in (("train_batch_nll", "#eb6834", "train batch"), ("val_nll", "#888780", "val"),
                              ("test_nll", "#2a78d6", "test")):  # fmt: skip
        m = [statistics.fmean(r[key] for r in by_step[s]) for s in steps]
        sd = [
            statistics.stdev([r[key] for r in by_step[s]]) if len(by_step[s]) > 1 else 0
            for s in steps
        ]
        ax[1].plot(x, m, marker="o", ms=3, color=color, label=f"{label} NLL")
        ax[1].fill_between(x, [max(a - b, 1e-6) for a, b in zip(m, sd)], [a + b for a, b in zip(m, sd)],
                           color=color, alpha=0.2)  # fmt: skip
    ax[0].set_ylim(97, 99.2)
    ax[0].set_ylabel("accuracy (%)")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("NLL")
    for a in ax:
        a.set_xlabel("epochs (samples seen / train size)")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle(f"{len(runs)} seed(s), mean ± std", fontsize=10)
    fig.tight_layout()
    fig.savefig(run_dir / "steps.png", dpi=150)
    return text


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(summarize(Path(sys.argv[1])))
