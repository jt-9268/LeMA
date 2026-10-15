import torch
import torchvision
import torch.distributed as dist
import torchvision.transforms as transforms
import torch.nn.functional as F
import lema
import argparse
import math
import os
import sys

from collections.abc import Callable, Sized
from torch import nn
from torch.utils.data import Dataset, Subset
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from typing import Any, cast, Optional
from loguru import logger

# Ensure corrent import
exp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if exp_dir not in sys.path:
    sys.path.insert(0, exp_dir)

from common.parser import (
    add_train_arguments,
    add_lema_arguments,
    add_profile_arguments,
)
from common.bench import measure_peak_memory, PeakMemoryMeasurement

TEST_INTERVAL = 1
VAL_SIZE = 5000
SPLIT_SEED = 0
SEED = 0
DAMP_START = 1e-3  # LeMA's own default
DAMP_MIN = 1e-1  # LeMA's own default (floor applies after the first accepted step)
LABEL_SMOOTHING = 0.0  # 0 = residual sqrt(NLL) exactly as before
NUM_CLASSES = 10
ECE_BINS = 15
ACTIVATION = "relu"  # relu + default init = the original network
INIT = "default"
OPTIMIZER = "lema"
LR = 1e-3  # AdamW only
WEIGHT_DECAY = (
    1e-2  # AdamW only, decoupled; applied to weight matrices/kernels, not biases
)
WARMUP_FRAC = 0.05  # AdamW only: linear warm-up fraction of all steps, then cosine to 0
DATA_DIR = ".data"

ACTIVATIONS = {"relu": F.relu, "elu": F.elu}  # elu: α = 1
# Variance-preserving gain g = 1/sqrt(E[f(z)²]), z ~ N(0, 1): relu √2; elu 1/sqrt(0.6449).
HIDDEN_GAIN = {"relu": math.sqrt(2.0), "elu": 1.2452}
LOG_PREFIX_FMT = "{time:YYYY-MM-DD HH:mm:ss.SSS}"


class Net(nn.Module):
    """Simple convolutional neural network.

    ``activation`` is applied after conv1, conv2 and fc1. With ``"relu"`` and the
    PyTorch default initialisation this is the original network.
    """

    def __init__(self, activation: str = ACTIVATION):
        super(Net, self).__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)}")
        self.activation = activation
        self._act = ACTIVATIONS[activation]
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self._act(x)
        x = self.conv2(x)
        x = self._act(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self._act(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


@torch.no_grad()
def init_he(model: Net) -> None:
    """He (variance-preserving) initialisation; call on CPU after seeding.

    Hidden layers (conv1, conv2, fc1): W ~ N(0, g²/fan_in), g = HIDDEN_GAIN[activation].
    Output layer (fc2): W ~ N(0, 1/fan_in). All biases zero. The PyTorch default
    W, b ~ U(±1/√fan_in) lets biases dominate the signal in narrow layers, which can
    switch whole ReLU layers off and leave LeMA with zero Jacobian columns.
    """
    gain = HIDDEN_GAIN[model.activation]
    for layer, g in (
        (model.conv1, gain),
        (model.conv2, gain),
        (model.fc1, gain),
        (model.fc2, 1.0),
    ):
        fan_in = layer.weight[0].numel()
        nn.init.normal_(layer.weight, 0.0, g / math.sqrt(fan_in))
        assert layer.bias is not None
        nn.init.zeros_(layer.bias)


def init_logger(rank: int, args):
    if rank != 0:
        return
    logger.remove()
    # Log into the standard output
    logger.add(
        sys.stdout,
        format=f"<green>{LOG_PREFIX_FMT}</green> | <level>{{message}}</level>",
        enqueue=True,
    )
    if args.log is not None:
        open(args.log, "w").close()
        # Log into the given file
        logger.add(args.log, format=f"{LOG_PREFIX_FMT} | {{message}}", enqueue=True)


def log_train(rank: int, result, epoch: int, batch_start: int):
    if rank != 0:
        return
    if isinstance(result, AdamWResult):
        logger.info(
            f"step info | epoch: {epoch} | batch: [{batch_start}, {batch_start + result.batch_size - 1}]"
            f" | loss: {result.loss:.3e} | method: adamw | iterations: 1 | lr: {result.lr:.3e}"
        )
        return
    step_method = "overdetermined" if result.overdetermined else "underdetermined"
    msg = "step info | epoch: {epoch} | batch: [{start}, {end}] | loss: {loss:.3e} | method: {method} | iterations: {iterations} | damp: {damp:.3e}".format(
        epoch=epoch,
        start=batch_start,
        end=batch_start + result.batch_size - 1,
        loss=result.loss,
        method=step_method,
        iterations=result.iterations,
        damp=result.damp,
    )
    logger.info(msg)


def log_epoch(rank: int, epoch: int, loss: float, mem_in_bytes: int):
    if rank != 0:
        return
    mem_in_gb = mem_in_bytes / 1e9
    msg = (
        "epoch info | epoch: {epoch} | loss: {loss:.3e} | memory: {mem:.2f} GB".format(
            epoch=epoch,
            loss=loss,
            mem=mem_in_gb,
        )
    )
    logger.info(msg)


def log_eval(rank: int, epoch: int, step: int, metrics: dict[str, tuple[float, float]]):
    """Log ``{split: (loss, accuracy)}`` for each held-out split on one line."""
    if rank != 0:
        return
    msg = f"eval info | epoch: {epoch} | step: {step}"
    for split, (loss, acc) in metrics.items():
        msg += f" | {split} loss: {loss:.3e} | {split} accuracy: {acc:.2%}"
    logger.info(msg)


def evaluate(
    model: nn.Module, testloader, device, keep: list | None = None
) -> tuple[float, float]:
    """Mean NLL and accuracy over the (sharded) split; if ``keep`` is a list, the local
    log-probabilities and labels are appended to it as CPU tensors."""
    model.eval()
    loss = torch.zeros((), device=device)
    corrects = torch.zeros((), dtype=torch.long, device=device)
    num_samples = torch.zeros((), dtype=torch.long, device=device)
    with torch.no_grad():
        for x, y in testloader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss += F.nll_loss(output, y, reduction="sum")
            corrects += (output.argmax(dim=1) == y).sum()
            num_samples += y.numel()
            if keep is not None:
                keep.append((output.float().cpu(), y.cpu()))
    dist.all_reduce(loss)
    dist.all_reduce(corrects)
    dist.all_reduce(num_samples)
    model.train()
    return loss.item() / num_samples.item(), corrects.item() / num_samples.item()


def gather_logprobs(local: list) -> tuple[torch.Tensor, torch.Tensor]:
    """All ranks' (log-probs, labels) for one split, concatenated (order irrelevant)."""
    lp = torch.cat([a for a, _ in local]) if local else torch.empty(0, NUM_CLASSES)
    y = torch.cat([b for _, b in local]) if local else torch.empty(0, dtype=torch.long)
    parts: list = [None] * dist.get_world_size()
    dist.all_gather_object(parts, (lp, y))
    return torch.cat([q[0] for q in parts]), torch.cat([q[1] for q in parts])


def calibration_metrics(
    logp: torch.Tensor,
    y: torch.Tensor,
    temperature: float = 1.0,
    n_bins: int = ECE_BINS,
) -> dict[str, float]:
    """NLL, accuracy, ECE and Brier score of softmax(logp / T), in float64.

    ``logp`` (N, C) may be log-probabilities or logits: softmax(z / T) is unchanged by
    a per-row shift, so log_softmax output works directly.
    ECE = sum_b (n_b / N) |acc_b - conf_b| over ``n_bins`` equal-width confidence bins
    (Guo et al. 2017); Brier = mean_i sum_c (p_ic - 1[c = y_i])^2, in [0, 2].
    """
    lp = torch.log_softmax(logp.double() / temperature, dim=1)
    p = lp.exp()
    conf, pred = p.max(dim=1)
    correct = (pred == y).double()
    bins = torch.clamp((conf * n_bins).long(), max=n_bins - 1)
    ece = torch.zeros((), dtype=torch.float64)
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.double().mean() * (correct[m].mean() - conf[m].mean()).abs()
    onehot = F.one_hot(y, logp.shape[1]).double()
    return {
        "nll": float(-lp.gather(1, y[:, None]).mean()),
        "acc": float(correct.mean()),
        "ece": float(ece),
        "brier": float(((p - onehot) ** 2).sum(dim=1).mean()),
        "conf": float(conf.mean()),
    }


def fit_temperature(
    logp: torch.Tensor,
    y: torch.Tensor,
    lo: float = 0.05,
    hi: float = 20.0,
    iters: int = 80,
) -> float:
    """T minimising NLL(softmax(logp / T)) by golden-section search on log T.

    NLL is convex in 1/T (log-sum-exp of functions linear in 1/T), hence unimodal in
    log T, so golden-section search converges to the minimiser within [lo, hi].
    """
    z, idx = logp.double(), y[:, None]

    def nll(log_t: float) -> float:
        return float(-torch.log_softmax(z / math.exp(log_t), 1).gather(1, idx).mean())

    a, b = math.log(lo), math.log(hi)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(d)
    return math.exp((a + b) / 2)


def shard_for_eval(dataset: Dataset) -> Subset:
    """Strided, unpadded shard of ``dataset`` for this rank.

    Unlike ``DistributedSampler``, this does not pad with duplicated samples when
    ``len(dataset) % world_size != 0``, so the all-reduced metrics in
    ``evaluate`` are exact over the full split.
    """
    n = len(cast(Sized, dataset))
    return Subset(dataset, range(dist.get_rank(), n, dist.get_world_size()))


def load_dataset(args):
    root_path = args.data_dir
    train_kwargs = {"batch_size": args.block_size}
    test_kwargs = {"batch_size": args.test_block_size}
    accel_kwargs = {"num_workers": 1, "persistent_workers": True, "pin_memory": True}

    train_kwargs.update(accel_kwargs)
    test_kwargs.update(accel_kwargs)
    # Load dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    fullset = torchvision.datasets.MNIST(
        root_path, train=True, download=False, transform=transform
    )
    testset = torchvision.datasets.MNIST(root_path, train=False, transform=transform)
    # Hold out a validation split from the official training set. The seeded
    # generator makes the split identical on every rank.
    if not 0 <= args.val_size < len(fullset):
        raise ValueError(
            f"--val-size must be in [0, {len(fullset)}), got {args.val_size}"
        )
    if args.val_size > 0:
        generator = torch.Generator().manual_seed(args.split_seed)
        trainset, valset = torch.utils.data.random_split(
            fullset, [len(fullset) - args.val_size, args.val_size], generator=generator
        )
    else:  # Identical training stream to the pre-validation code path
        trainset, valset = fullset, Subset(fullset, [])
    if dist.get_rank() == 0:
        logger.info(
            f"Train / val / test samples: "
            f"{len(trainset):,} / {len(valset):,} / {len(testset):,}"
        )
    # Create distributed sampler for training
    # --seed also sets the shuffle order (seed 0 reproduces the earlier fixed order)
    trainsampler = DistributedSampler(trainset, seed=args.seed)
    # Create loaders
    trainloader = torch.utils.data.DataLoader(
        trainset, sampler=trainsampler, **train_kwargs
    )
    valloader = (
        torch.utils.data.DataLoader(shard_for_eval(valset), **test_kwargs)
        if args.val_size > 0
        else None
    )
    testloader = torch.utils.data.DataLoader(shard_for_eval(testset), **test_kwargs)
    return trainloader, valloader, testloader


def evaluate_splits(
    model: nn.Module,
    loaders: dict[str, torch.utils.data.DataLoader | None],
    device,
    keep: dict[str, list] | None = None,
) -> dict[str, tuple[float, float]]:
    """``{split: (mean NLL, accuracy)}`` over every non-empty held-out split.

    If ``keep`` is a dict, ``keep[split]`` collects this rank's (log-probs, labels).
    """
    out = {}
    for split, loader in loaders.items():
        if loader is None:
            continue
        local = keep.setdefault(split, []) if keep is not None else None
        out[split] = evaluate(model, loader, device, keep=local)
    return out


def log_calibration(rank: int, epoch: int, step: int, keep: dict[str, list]) -> None:
    """Fit T on validation; log test NLL/ECE/Brier before and after scaling (rank 0).

    Collective: every rank must call it (gathers log-probs from all ranks).
    """
    if "val" not in keep or "test" not in keep:
        return
    val = gather_logprobs(keep["val"])
    test = gather_logprobs(keep["test"])
    if rank != 0:
        return
    t = fit_temperature(*val)
    v1, t1 = calibration_metrics(*val), calibration_metrics(*test)
    vt, tt = (
        calibration_metrics(*val, temperature=t),
        calibration_metrics(*test, temperature=t),
    )
    logger.info(
        f"calib info | epoch: {epoch} | step: {step} | temperature: {t:.4f}"
        f" | val ece: {v1['ece']:.3%} | val ece@T: {vt['ece']:.3%}"
        f" | test nll: {t1['nll']:.4e} | test nll@T: {tt['nll']:.4e}"
        f" | test ece: {t1['ece']:.3%} | test ece@T: {tt['ece']:.3%}"
        f" | test brier: {t1['brier']:.4e} | test brier@T: {tt['brier']:.4e}"
        f" | test conf: {t1['conf']:.3%} | test conf@T: {tt['conf']:.3%}"
    )


class AdamWStep:
    """One data-parallel AdamW step with a LeMA-like result for logging.

    Loss is the mean NLL over the local block; DDP averages gradients over ranks, so the
    update uses the global-batch mean gradient. ``loss`` reported is the global mean NLL
    of the batch *before* the update (LeMA reports it after its update).
    """

    def __init__(
        self, model: nn.Module, args, steps_total: int, device, loss_fn=None
    ) -> None:
        self.loss_fn = loss_fn or (lambda out, y: F.nll_loss(out, y))
        self.ddp = DistributedDataParallel(
            model, device_ids=[device] if torch.cuda.is_available() else None
        )  # broadcasts rank 0's initial weights
        decay = [p for p in model.parameters() if p.ndim > 1]
        no_decay = [p for p in model.parameters() if p.ndim <= 1]
        self.opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": args.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=args.lr,
        )  # fmt: skip
        warm = max(1, round(args.warmup_frac * steps_total))

        def lr_lambda(step: int) -> float:  # linear warm-up, then cosine to 0
            if step < warm:
                return (step + 1) / warm
            t = (step - warm) / max(1, steps_total - warm)
            return 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))

        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)

    def step(self, x: torch.Tensor, y: torch.Tensor) -> "AdamWResult":
        self.opt.zero_grad(set_to_none=True)
        loss = self.loss_fn(self.ddp(x), y)
        loss.backward()
        lr = self.opt.param_groups[0]["lr"]
        self.opt.step()
        self.sched.step()
        stats = torch.stack(
            [loss.detach() * len(y), torch.tensor(float(len(y)), device=loss.device)]
        )
        dist.all_reduce(stats)
        return AdamWResult(
            loss=(stats[0] / stats[1]).item(), batch_size=int(stats[1].item()), lr=lr
        )


class AdamWResult:
    def __init__(self, loss: float, batch_size: int, lr: float) -> None:
        self.loss, self.batch_size, self.lr = loss, batch_size, lr
        self.iterations, self.terminate = 1, False


def make_residual_fn(eps: float, num_classes: int = NUM_CLASSES):
    """Per-sample residual r_i with sum_i r_i^2 the training loss LeMA minimises.

    eps = 0: r = sqrt(NLL + 1e-9), the original residual (returned unchanged).
    eps > 0: r = sqrt(KL(y~ || p) + 1e-9) with y~ = (1 - eps) on the true class and
    eps / (C - 1) on each other class. KL = 0 only at p = y~, i.e. at the finite logit
    margin m* = ln[(1 - eps)(C - 1) / eps], so Gauss-Newton steps converge to m*
    instead of growing the margin without bound (as they do for sqrt(NLL)).
    """
    if eps == 0.0:
        return residual_fn
    if not 0.0 < eps < 1.0 - 1.0 / num_classes:
        raise ValueError("--label-smoothing must be in (0, 1 - 1/C)")
    on, off = 1.0 - eps, eps / (num_classes - 1)
    neg_entropy = on * math.log(on) + (num_classes - 1) * off * math.log(off)

    def smoothed_residual(logp, y):
        logp_y = logp.gather(1, y[:, None]).squeeze(1)
        cross = -(off * logp.sum(dim=1) + (on - off) * logp_y)  # -sum_c y~_c log p_c
        return torch.sqrt((cross + neg_entropy).clamp_min(0.0) + 1e-9)

    return smoothed_residual


def residual_fn(a, b):
    return torch.sqrt(F.nll_loss(a, b, reduction="none") + 1e-9)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser = add_lema_arguments(
        parser,
        block_size=1024,
        shard_size=128,
        slice_size=32,
        damp_ratio=10.0,
    )
    parser = add_train_arguments(
        parser,
        epochs=10,
        test_block_size=1024,
    )
    parser = add_profile_arguments(
        parser,
        profile_steps=5,
    )
    parser.add_argument(
        "--test-interval",
        type=int,
        default=TEST_INTERVAL,
        metavar="N",
        help=f"Interval of test (in steps)",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=VAL_SIZE,
        metavar="N",
        help="Number of training samples held out for validation (0 disables)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=SPLIT_SEED,
        metavar="N",
        help="Seed for the train/validation split",
    )
    parser.add_argument(
        "--activation",
        type=str,
        choices=sorted(ACTIVATIONS),
        default=ACTIVATION,
        help="Hidden-layer activation (relu reproduces the original network)",
    )
    parser.add_argument(
        "--init",
        type=str,
        choices=["default", "he"],
        default=INIT,
        help="default: PyTorch layer init; he: N(0, g²/fan_in) weights, zero biases",
    )
    parser.add_argument(
        "--damp-start",
        type=float,
        default=DAMP_START,
        metavar="X",
        help="LeMA damping tried on the first step (e.g. 1e4 forces a damped warm-up)",
    )
    parser.add_argument(
        "--damp-min",
        type=float,
        default=DAMP_MIN,
        metavar="X",
        help="LeMA damping floor (lambda never decreases below it after an accepted step)",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=LABEL_SMOOTHING,
        metavar="EPS",
        help="Train on KL to targets with 1-EPS on the true class (LeMA residual sqrt(KL)); 0 = NLL",
    )
    parser.add_argument(
        "--calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="At each evaluation fit a temperature on val; log test NLL/ECE/Brier before and after",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["lema", "adamw"],
        default=OPTIMIZER,
        help="lema, or the AdamW baseline (uses --block-size as per-GPU batch)",
    )
    parser.add_argument(
        "--lr", type=float, default=LR, metavar="X", help="AdamW peak learning rate"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=WEIGHT_DECAY, metavar="X",
        help="AdamW decoupled weight decay (weights only, not biases)",
    )  # fmt: skip
    parser.add_argument(
        "--warmup-frac", type=float, default=WARMUP_FRAC, metavar="X",
        help="AdamW linear warm-up as a fraction of all steps; cosine decay to 0 after",
    )  # fmt: skip
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        metavar="N",
        help="Seed for model initialisation and training-data shuffling",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DATA_DIR,
        metavar="FILE",
        help=f"Dataset directory",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        metavar="FILE",
        help=f"Log file path (default: None)",
    )
    args = parser.parse_args()

    # Initialize process
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # Initialize logger
    init_logger(rank, args)

    # Load dataset
    trainloader, valloader, testloader = load_dataset(args)
    evalloaders = {"val": valloader, "test": testloader}

    # Initialize the model
    device = torch.cuda.current_device()
    torch.manual_seed(args.seed)  # draws on CPU → independent of device and rank
    model = Net(args.activation)
    if args.init == "he":
        init_he(model)
    model = model.to(device)
    batch_size = args.block_size * dist.get_world_size()
    model_size = sum(p.numel() for p in model.parameters())
    mem_msgs: list[str] = []
    step_fn: Callable[[torch.Tensor, torch.Tensor], Any]
    if args.optimizer == "lema":
        optim = lema.LeMA(
            model=model,
            residual_fn=make_residual_fn(args.label_smoothing),
            damp_ratio=args.damp_ratio,
            damp_start=args.damp_start,
            damp_min=args.damp_min,
        )
        element_size = torch.tensor([], dtype=optim._optim_dtype).element_size()
        hessian_bytes = min(batch_size**2, model_size**2) * element_size
        jacobian_shard_bytes = args.shard_size * model_size * element_size
        mem_msgs = [
            f"Hessian memory: {hessian_bytes / 1e9:.3f} GB",
            f"Jacobian shard memory: {jacobian_shard_bytes / 1e9:.3f} GB",
        ]

        def lema_step(x, y):
            return optim.step(
                x, y, shard_size=args.shard_size, slice_size=args.slice_size
            )

        step_fn = lema_step
    else:
        smoothed = make_residual_fn(args.label_smoothing)
        adamw = AdamWStep(
            model, args, steps_total=args.epochs * len(trainloader), device=device,
            loss_fn=None if args.label_smoothing == 0.0
            else (lambda out, y: smoothed(out, y).square().mean()),
        )  # fmt: skip
        step_fn = adamw.step
        mem_msgs = [
            (
                f"Optimizer: adamw | lr: {args.lr:g} | weight decay: {args.weight_decay:g}"
                f" | warmup: {args.warmup_frac:g} | steps: {args.epochs * len(trainloader)}"
                f" | label smoothing: {args.label_smoothing:g}"
            )
        ]

    # Log launch information and estimated memory usage
    if rank == 0:
        header = (
            f"Activation: {args.activation} | init: {args.init} | seed: {args.seed}"
        )
        if args.optimizer == "lema":
            header += (
                f" | damp start: {args.damp_start:g} | damp min: {args.damp_min:g}"
                f" | label smoothing: {args.label_smoothing:g}"
            )
        logger.info(header)
        logger.info(f"Model size: {model_size:,}")
        logger.info(f"Batch size: {batch_size:,}")
        for msg in mem_msgs:
            logger.info(msg)

    # Train
    nsteps = 0
    epoch = 0
    last_eval_step = -1
    terminate = False
    for epoch in range(1, args.epochs + 1):
        cast(DistributedSampler, trainloader.sampler).set_epoch(epoch - 1)
        loss = 0.0
        num_samples = 0
        batch_start = 1
        with measure_peak_memory(device=device) as mem:  # Measure peak memory usage
            n_batches = len(trainloader)
            for batch_idx, (x, y) in enumerate(trainloader, start=1):
                x = x.to(device)
                y = y.to(device)
                res = step_fn(x, y)
                loss += res.loss * res.batch_size
                num_samples += res.batch_size
                terminate = res.terminate
                log_train(rank=rank, result=res, epoch=epoch, batch_start=batch_start)
                batch_start += res.batch_size
                nsteps += 1
                # Evaluate on held-out validation and test splits
                if nsteps % args.test_interval == 0 or batch_idx == n_batches:
                    keep: dict[str, list] | None = {} if args.calibration else None
                    metrics = evaluate_splits(model, evalloaders, device, keep=keep)
                    log_eval(rank=rank, epoch=epoch, step=nsteps, metrics=metrics)
                    if keep is not None:
                        log_calibration(rank, epoch, nsteps, keep)
                    last_eval_step = nsteps
                if args.profile and nsteps >= args.profile_steps:
                    terminate = True
                if terminate:
                    break
        log_epoch(
            rank=rank,
            epoch=epoch,
            loss=loss / max(num_samples, 1),
            mem_in_bytes=mem.peak_bytes,
        )
        if terminate:
            break

    # Always report held-out metrics for the final parameters
    if nsteps != last_eval_step:
        keep = {} if args.calibration else None
        metrics = evaluate_splits(model, evalloaders, device, keep=keep)
        log_eval(rank=rank, epoch=epoch, step=nsteps, metrics=metrics)
        if keep is not None:
            log_calibration(rank, epoch, nsteps, keep)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
