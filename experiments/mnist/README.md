# Train a Convolutional Neural Network on MNIST with LeMA

`train.py` trains a small CNN on MNIST with the LeMA multi-GPU Levenberg–Marquardt
optimizer, launched with `torchrun` (one process per GPU). It reports loss and accuracy on a
held-out validation split and on the official test set, and, by default, calibration
metrics (ECE, Brier score and temperature-scaled NLL) at every evaluation.

## Quick start

Recommended settings on one node with 4 GPUs (global batch 4 × 2,048 = 8,192):

```bash
bash scripts/download_mnist.sh            # writes .data/MNIST/raw/ under the current directory
torchrun --standalone --nproc_per_node 4 experiments/mnist/train.py \
    --block-size 2048 --shard-size 2048 --slice-size 1024 \
    --label-smoothing 0.01 --epochs 5 2>&1 | tee mnist.log
```

Without `--label-smoothing` the script trains the original network on the original loss.
The section on calibration explains why smoothing helps here.

## Model

| Layer | Operation | Output shape | Parameters |
| --- | --- | --- | --- |
| conv1 | Conv 1→32, 3×3, activation | (B, 32, 26, 26) | 320 |
| conv2 | Conv 32→64, 3×3, activation, 2×2 max-pool | (B, 64, 12, 12) | 18,496 |
| fc1 | Linear 9216→128, activation | (B, 128) | 1,179,776 |
| fc2 | Linear 128→10, log-softmax | (B, 10) | 1,290 |
| total | | | 1,199,882 |

The defaults (`--activation relu --init default`) are the original network with PyTorch's
layer initialisation, W, b ~ U(±1/√fan_in). The weights are drawn on the CPU under
`--seed` and broadcast from rank 0, so a given seed starts from the same weights on any
device or GPU count. `--seed` also sets the training-data order, so runs that differ in
one option but share a seed see the same batches.

Alternatives are available:

- **`--activation elu`:** ELU (α = 1). Its derivative is never zero, which prevents dead
  units in much narrower variants of this network. At this width it trains to the same
  accuracy as ReLU, but each step is about 5% slower.
- **`--init he`:** hidden-layer weights drawn from N(0, g²/fan_in), with
  g = 1/√E[f(z)²] for z ~ N(0, 1) (√2 for ReLU, 1.2452 for ELU), output layer g = 1, and
  zero biases. This gives a faster first epoch, but in 5-seed runs at this width it ended
  about 0.1 percentage points below the default initialisation.

## Loss, label smoothing and calibration

LeMA minimises Σᵢ rᵢ². By default rᵢ = √(NLLᵢ + 10⁻⁹), so Σᵢ rᵢ² is the summed NLL.

**This residual drives overconfidence.** For a correctly classified sample with logit
margin m, NLL ≈ e^(−m), so r ≈ e^(−m/2). A Gauss–Newton step on that residual then changes
the margin by δm = −r/(dr/dm) = 2, *regardless of how small the loss already is*,
because r = 0 is reached only at infinite margin. Over training, margins keep growing:
the training NLL collapses towards 10⁻⁴, and validation NLL rises after about one epoch
while accuracy stays flat.

**`--label-smoothing ε` removes the cause.** The residual becomes
r = √(KL(ỹ‖p) + 10⁻⁹), where the target ỹ puts 1−ε on the true class and ε/(C−1) on each
other class. KL is zero at the *finite* margin m* = ln[(1−ε)(C−1)/ε], so Gauss–Newton
converges to m* (6.8 nats for ε = 0.01) instead of growing the margin without bound.
Choose ε close to the expected error rate. Larger values cap training confidence below
the accuracy and make the model underconfident.

**Calibration is logged at every evaluation** (`--no-calibration` turns it off). A
temperature T is fitted on the validation set: NLL is convex in 1/T, so a golden-section
search on log T finds the minimum. The `calib info` line then reports test NLL, ECE
(15 equal-width bins), Brier score and mean confidence, before and after dividing the
logits by T. Temperature scaling never changes predictions, so accuracy is unaffected.

**Results.** These are 5 seeds, global batch 8,192, 5 epochs, with ReLU and default
initialisation. Metrics are means at the end of epoch 5, ± the standard deviation over
seeds.

| Configuration | Hardware | Test accuracy | Test NLL | Test ECE | ECE / NLL after T |
| --- | --- | --- | --- | --- | --- |
| LeMA, ε = 0 (default) | 4 × H200 | 99.04 ± 0.03 % | 0.0557 | not logged¹ | not logged¹ |
| LeMA, ε = 0.01 | 4 × A100-40GB² | 99.17 ± 0.04 % | 0.0477 | 2.36 % | 0.21 % / 0.0279 (T = 0.63) |
| AdamW, lr 6×10⁻³, batch 512 | 4 × H200 | 99.12 ± 0.05 % | 0.0285 | not logged¹ | not logged¹ |

¹ These runs predate calibration logging.
² Run with `--shard-size 1024 --slice-size 512` to fit in 40 GB.

Paired by seed (same initial weights and batches), ε = 0.01 improves test accuracy by
0.13 percentage points over ε = 0 (5/5 seeds) and stops the post-minimum rise in
validation NLL. The two rows ran on different hardware, which adds floating-point
differences of a few hundredths of a percentage point. With ε = 0.01 the raw model is
slightly *under*confident: mean confidence is 96.8% at 99.2% accuracy, and the fitted
T = 0.63 corrects it.

## Data

The data is split as follows:

- **Validation:** `--val-size` images (default 5,000) are held out from the 60,000
  training images using a fixed seed (`--split-seed`). The split is identical on every rank.
- **Training:** the remaining images. `--val-size 0` disables the hold-out and trains on
  all 60,000.
- **Test:** the official 10,000-image test set.
- **Evaluation:** validation and test sets are sharded across ranks without padding, so
  the reported metrics are exact for any number of GPUs.

`--data-dir` must point at the directory that contains `MNIST/raw/` with the four
uncompressed `*-ubyte` files (default `.data`, relative to the current directory).
Compute nodes on clusters often have no internet access, so download the data on a login
node first.

Inputs are normalised with the standard MNIST constants (mean 0.1307, std 0.3081). These
were computed over all 60,000 training images, so they include the validation images;
for MNIST the effect is negligible.

## Performance notes

Two settings dominate cost and memory:

- **Shard size:** per step, each GPU evaluates 2b + (b/s)·B Jacobian rows, for block size
  b, shard size s and global batch B. Setting `--shard-size` equal to `--block-size`
  avoids recomputing the column Jacobians and is the fastest setting.
- **Memory:** peak memory per GPU is about 5 × shard × 1.2M parameters × 4 bytes: about
  50 GB for shard 2,048 and 25 GB for shard 1,024. On 40 GB GPUs use
  `--shard-size 1024 --slice-size 512`. These figures were measured with LeMA at commit
  8177ef8; later commits evaluate the loss check and the update in `--slice-size` chunks,
  which may lower them.

At this model size the step time is dominated by forming the B × B Gram matrix. On an
H200 a step at B = 8,192 takes about 1.4 s, and about 4 s on an A100.

`--log FILE` cannot be used under `torchrun`. `torchrun` rejects `--log` as an ambiguous
prefix of its own `--log-dir`/`--logs-specs` options, so capture stdout with `tee` instead.

## Multi-seed experiments

For several seeds, run the script once per seed and `tee` each run's output into a
directory of logs:

```bash
mkdir -p runs/ls0.01/logs
for s in 0 1 2 3 4; do
  torchrun --standalone --nproc_per_node 4 experiments/mnist/train.py \
      --block-size 2048 --shard-size 2048 --slice-size 1024 \
      --label-smoothing 0.01 --epochs 5 --seed $s 2>&1 | tee runs/ls0.01/logs/seed$s.log
done
```

Then summarise or compare the runs:

- **`summarize_seeds.py RUN_DIR`** writes `summary.md` and `steps.csv`, and `steps.png` if
  matplotlib is installed. They contain, for each seed, the best-validation-NLL and
  best-validation-accuracy selections and the step time; mean ± std at every epoch end and
  every step; and the calibration tables.
- **`compare_runs.py RUN_A RUN_B --labels A B`** writes `compare_vs_<A>.md`. It pairs the
  two runs by seed and reports paired differences, matched-epoch and matched-wall-clock
  comparisons, and the time each run takes to reach target validation accuracies.

## Command-line arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--block-size` | `int` | `1024` | Training samples per process (per-GPU batch for AdamW); global batch = block size × number of GPUs. |
| `--shard-size` | `int` | `128` | Shard size in LeMA's Jacobian/Gram computations; equal to `--block-size` is fastest (see above). |
| `--slice-size` | `int` | `32` | Chunk size for per-sample Jacobian evaluation (vmap). |
| `--damp-ratio` | `float` | `10.0` | Factor by which LeMA's damping is raised after a rejected trial and lowered after an accepted step. |
| `--damp-start` | `float` | `1e-3` | Damping tried on the first step. A large value (e.g. `1e4`) forces a damped, gradient-like warm-up. |
| `--damp-min` | `float` | `0.1` | Damping floor. After an accepted step λ is divided by `--damp-ratio` but never below this value (the first trial uses `--damp-start` unclamped). |
| `--epochs` | `int` | `10` | Number of training epochs. |
| `--test-block-size` | `int` | `1024` | Batch size for validation/test evaluation. |
| `--test-interval` | `int` | `1` | Training steps between evaluations; the last step of every epoch and the final parameters are always evaluated. |
| `--val-size` | `int` | `5000` | Images held out from the training set for validation; `0` disables. |
| `--split-seed` | `int` | `0` | Seed for the train/validation split. |
| `--seed` | `int` | `0` | Seed for model initialisation and training-data order. |
| `--activation` | `str` | `relu` | Hidden activation: `relu` or `elu`. |
| `--init` | `str` | `default` | `default`: PyTorch layer initialisation; `he`: N(0, g²/fan_in) weights, zero biases. |
| `--label-smoothing` | `float` | `0` | Train on KL(ỹ‖p) with 1−ε on the true class (LeMA residual √KL); `0` = the original √NLL residual. |
| `--calibration` / `--no-calibration` | flag | on | Fit a temperature on validation at each evaluation and log test NLL/ECE/Brier/confidence before and after. Training is unaffected. |
| `--optimizer` | `str` | `lema` | `lema`, or the `adamw` baseline (LeMA-only options are ignored). |
| `--lr` | `float` | `1e-3` | AdamW peak learning rate. |
| `--weight-decay` | `float` | `0.01` | AdamW decoupled weight decay, applied to weight matrices and kernels only (not biases). |
| `--warmup-frac` | `float` | `0.05` | AdamW linear warm-up as a fraction of all steps, then cosine decay to 0. |
| `--data-dir` | `str` | `.data` | Directory containing `MNIST/raw/`. |
| `--log` | `str` | `None` | Log file path (unusable under `torchrun`; see above). |
| `--profile` | flag | off | Enable profiling mode. |
| `--profile-steps` | `int` | `5` | Steps to run before leaving profiling mode. |

## Output

At start-up, rank 0 logs:

- the train/validation/test sizes;
- the configuration (activation, initialisation, seed, damping, label smoothing);
- the model and batch size;
- for LeMA, the estimated memory for the Gram matrix and for one Jacobian shard.

During training it logs these lines:

```
step info  | epoch: E | batch: [first, last] | loss: L | method: underdetermined | iterations: I | damp: λ
eval info  | epoch: E | step: S | val loss: … | val accuracy: …% | test loss: … | test accuracy: …%
calib info | epoch: E | step: S | temperature: T | val ece: …% | val ece@T: …% | test nll: … | test nll@T: … | test ece: …% | test ece@T: …% | test brier: … | test brier@T: … | test conf: …% | test conf@T: …%
epoch info | epoch: E | loss: L | memory: M GB
```

The fields mean:

- **`loss` in a step line** is the mean training loss on that step's batch after the
  update. That is the NLL, or the KL to the smoothed targets when `--label-smoothing > 0`.
  For AdamW it is measured before the update, and the `damp` field is replaced by `lr`.
- **`iterations`** is the number of damping trials before a step was accepted (at most 10).
- **`damp`** is the damping that will be tried first on the next step.
- **`val loss` and `test loss` in an eval line** are always the plain NLL, whatever the
  training loss.
