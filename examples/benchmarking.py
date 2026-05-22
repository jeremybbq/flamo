import argparse
import json
import os
import random
import time
from collections import OrderedDict

import numpy as np
import torch
from flamo.optimize.dataset import DatasetColorless
from flamo.optimize.trainer import Trainer
from flamo.processor import dsp, system
from flamo.optimize.loss import sparsity_loss, masked_mse_loss
from flamo.functional import skew_matrix
from flamo.utils import save_audio


DEFAULT_SEED = 130709


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def primes_in_range(low: int, high: int) -> list[int]:
    if high < 2 or high < low:
        return []
    low = max(low, 2)
    sieve = [True] * (high + 1)
    sieve[0:2] = [False, False]
    for i in range(2, int(high**0.5) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start : high + 1 : step] = [False] * len(range(start, high + 1, step))
    return [p for p in range(low, high + 1) if sieve[p]]


def sample_coprime_delays(n: int, low: int, high: int) -> torch.Tensor:
    primes = primes_in_range(low, high)
    if len(primes) < n:
        raise ValueError(
            f"Not enough coprime candidates in [{low}, {high}] to sample {n} delays."
        )
    delays = random.sample(primes, n)
    return torch.tensor(delays, dtype=torch.int64)


def hadamard_power2(n: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    U = torch.tensor([[1.0]], device=device, dtype=dtype)
    while U.shape[0] < n:
        U = torch.kron(
            U, torch.tensor([[1, 1], [1, -1]], device=device, dtype=dtype)
        ) / torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype))
    return U


def build_fdn(
    args: argparse.Namespace,
    N: int,
    delay_lengths: torch.Tensor,
    feedback_kind: str,
) -> tuple[system.Shell, dict]:
    alias_decay_db = args.alias_decay_db
    delay_lengths = delay_lengths.to(torch.int64)

    input_gain = dsp.Gain(
        size=(N, 1),
        nfft=args.nfft,
        requires_grad=True,
        alias_decay_db=alias_decay_db,
        device=args.device,
        dtype=args.dtype,
    )
    output_gain = dsp.Gain(
        size=(1, N),
        nfft=args.nfft,
        requires_grad=True,
        alias_decay_db=alias_decay_db,
        device=args.device,
        dtype=args.dtype,
    )

    delays = dsp.parallelDelay(
        size=(N,),
        max_len=int(delay_lengths.max().item()),
        nfft=args.nfft,
        isint=True,
        requires_grad=False,
        alias_decay_db=alias_decay_db,
        device=args.device,
        dtype=args.dtype,
    )
    delays.assign_value(delays.sample2s(delay_lengths))

    metadata: dict = {}
    if feedback_kind == "baseline":
        feedback = dsp.Matrix(
            size=(N, N),
            nfft=args.nfft,
            matrix_type="orthogonal",
            requires_grad=False,
            alias_decay_db=alias_decay_db,
            device=args.device,
            dtype=args.dtype,
        )
        with torch.no_grad():
            feedback.param.data = torch.randn(
                N, N, device=args.device, dtype=args.dtype
            )
    elif feedback_kind == "hadamard":
        if (N & (N - 1)) != 0:
            raise ValueError(
                f"Hadamard baseline requires power-of-two N, got N={N}."
            )
        feedback = dsp.Matrix(
            size=(N, N),
            nfft=args.nfft,
            matrix_type="hadamard",
            requires_grad=False,
            alias_decay_db=alias_decay_db,
            device=args.device,
            dtype=args.dtype,
        )
    elif feedback_kind == "orthogonal":
        feedback = dsp.Matrix(
            size=(N, N),
            nfft=args.nfft,
            matrix_type="orthogonal",
            requires_grad=True,
            alias_decay_db=alias_decay_db,
            device=args.device,
            dtype=args.dtype,
        )
    elif feedback_kind == "householder":
        feedback = dsp.HouseholderMatrix(
            size=(N, N),
            nfft=args.nfft,
            requires_grad=True,
            alias_decay_db=alias_decay_db,
            device=args.device,
            dtype=args.dtype,
        )
    elif feedback_kind == "multi_householder":
        feedback = dsp.MultiHouseholderMatrix(
            size=(N, N),
            num_reflections=args.num_reflections,
            nfft=args.nfft,
            requires_grad=True,
            alias_decay_db=alias_decay_db,
            device=args.device,
            dtype=args.dtype,
        )
    elif feedback_kind in ("scattering", "scattering_hadamard", "scattering_householder"):
        min_delay = int(delay_lengths.min().item())
        high = max(2, int(np.floor(min_delay / 2)))
        m_L = torch.randint(
            low=1,
            high=high,
            size=[N],
            device=args.device,
            dtype=args.dtype,
        )
        m_R = torch.randint(
            low=1,
            high=high,
            size=[N],
            device=args.device,
            dtype=args.dtype,
        )
        feedback = dsp.ScatteringMatrix(
            size=(4, N, N),
            nfft=args.nfft,
            gain_per_sample=1,
            sparsity=3,
            m_L=m_L,
            m_R=m_R,
            alias_decay_db=alias_decay_db,
            requires_grad=True,
            device=args.device,
            dtype=args.dtype,
        )
        metadata["m_L"] = m_L
        metadata["m_R"] = m_R

        if feedback_kind == "scattering_hadamard":
            if (N & (N - 1)) != 0:
                raise ValueError(
                    f"Hadamard scattering requires power-of-two N, got N={N}."
                )
            fixed_h = hadamard_power2(N, device=args.device, dtype=args.dtype)

            def map_with_fixed(param: torch.Tensor) -> torch.Tensor:
                U_last = torch.matrix_exp(skew_matrix(param[-1]))
                fixed = fixed_h.to(param.device, param.dtype)
                U = [fixed] * (param.shape[0] - 1) + [U_last]
                return torch.stack(U, dim=0)

            feedback.map = map_with_fixed
        elif feedback_kind == "scattering_householder":

            def map_householder(param: torch.Tensor) -> torch.Tensor:
                # param shape: (K, N, N); build K Householder matrices of size N x N
                # Use first column of each stage as Householder vector.
                u = param[..., 0]  # shape (K, N)
                norm = torch.norm(u, dim=-1, keepdim=True)
                norm = torch.clamp(norm, min=torch.finfo(param.dtype).eps)
                u_unit = u / norm
                eye = torch.eye(
                    param.shape[-1], device=param.device, dtype=param.dtype
                ).unsqueeze(0)
                return eye - 2.0 * u_unit.unsqueeze(-1) * u_unit.unsqueeze(-2)

            feedback.map = map_householder
    else:
        raise ValueError(f"Unsupported feedback kind: {feedback_kind}")

    feedback_loop = system.Recursion(fF=delays, fB=feedback)
    FDN = system.Series(
        OrderedDict(
            {
                "input_gain": input_gain,
                "feedback_loop": feedback_loop,
                "output_gain": output_gain,
            }
        )
    )

    input_layer = dsp.FFT(args.nfft, dtype=args.dtype)
    output_layer = dsp.Transform(transform=lambda x: torch.abs(x), dtype=args.dtype)
    model = system.Shell(core=FDN, input_layer=input_layer, output_layer=output_layer)

    return model, metadata


def collect_params(model: system.Shell) -> dict:
    core = model.get_core()
    params: dict[str, np.ndarray] = {}
    params["input_gain"] = core.input_gain.param.detach().cpu().numpy()
    params["output_gain"] = core.output_gain.param.detach().cpu().numpy()
    params["delay_param"] = core.feedback_loop.feedforward.param.detach().cpu().numpy()
    params["delay_samples"] = (
        core.feedback_loop.feedforward.s2sample(
            core.feedback_loop.feedforward.map(core.feedback_loop.feedforward.param)
        )
        .detach()
        .cpu()
        .numpy()
    )
    feedback = core.feedback_loop.feedback
    params["feedback_param"] = feedback.param.detach().cpu().numpy()
    try:
        params["feedback_mapped"] = (
            feedback.map(feedback.param).detach().cpu().numpy()
        )
    except Exception:
        pass
    if hasattr(feedback, "map_filter"):
        params["scattering_shifts"] = (
            feedback.map_filter.shifts.detach().cpu().numpy()
        )
        if getattr(feedback, "m_L", None) is not None:
            params["m_L"] = feedback.m_L.detach().cpu().numpy()
        if getattr(feedback, "m_R", None) is not None:
            params["m_R"] = feedback.m_R.detach().cpu().numpy()
    return params


def save_config(path: str, config: dict) -> None:
    def _no_complex(obj):
        """Recursively convert for JSON: complex with imag=0 -> real, else [real, imag]."""
        if isinstance(obj, (list, tuple)):
            return [_no_complex(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _no_complex(v) for k, v in obj.items()}
        if isinstance(obj, (complex, np.complexfloating)):
            r, i = float(obj.real), float(obj.imag)
            if abs(i) < 1e-15:
                return r
            return [r, i]
        return obj

    def to_jsonable(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            return value.item()
        elif isinstance(value, (torch.dtype, np.dtype)):
            return str(value)
        elif isinstance(value, torch.device):
            return str(value)
        else:
            return _no_complex(value)
        return _no_complex(value)

    config = {k: to_jsonable(v) for k, v in config.items()}
    with open(path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def run_training(
    args: argparse.Namespace,
    run_dir: str,
    model: system.Shell,
    dataset_expand: int,
    extra_config: dict,
) -> None:
    os.makedirs(run_dir, exist_ok=True)

    dataset = DatasetColorless(
        input_shape=(1, args.nfft // 2 + 1, 1),
        target_shape=(1, args.nfft // 2 + 1, 1),
        expand=dataset_expand,
        device=args.device,
        dtype=args.dtype,
    )
    train_loader, valid_loader = load_dataset_seeded(
        dataset,
        batch_size=args.batch_size,
        split=args.split,
        shuffle=True,
        seed=args.seed,
        device=args.device,
    )

    trainer = Trainer(
        model,
        max_epochs=args.max_epochs,
        lr=args.lr,
        patience=args.patience,
        patience_delta=args.patience_delta,
        train_dir=run_dir,
        device=args.device,
        log=False,
    )
    trainer.register_criterion(
        masked_mse_loss(
            nfft=args.nfft,
            n_samples=args.mask_samples,
            n_sets=1,
            regenerate_mask=True,
            device=args.device,
        ),
        1,
    )
    trainer.register_criterion(sparsity_loss(), args.sparsity_weight, requires_model=True)

    start_time = time.time()
    trainer.train(train_loader, valid_loader)
    train_time = time.time() - start_time

    # Save optimized impulse response as audio for inference inspection
    with torch.no_grad():
        ir_optim = model.get_time_response(
            identity=False, fs=args.samplerate
        ).squeeze()
        peak = torch.max(torch.abs(ir_optim))
        if peak > 0:
            ir_optim = ir_optim / peak
        ir_path = os.path.join(run_dir, "ir_optim.wav")
        save_audio(ir_path, ir_optim, fs=args.samplerate)
        print(f"  Saved {ir_path}")

    params = collect_params(model)
    params_npz_path = os.path.join(run_dir, "params.npz")
    np.savez(params_npz_path, **params)
    print(f"  Saved {params_npz_path}")

    # Also store parameters as JSON for easier inspection
    params_json_path = os.path.join(run_dir, "params.json")
    save_config(params_json_path, params)
    print(f"  Saved {params_json_path}")

    model_path = os.path.join(run_dir, "model_state.pt")
    torch.save(model.state_dict(), model_path)
    print(f"  Saved {model_path}")

    loss_log = {
        "train_loss": trainer.train_loss,
        "valid_loss": trainer.valid_loss,
        "train_time_sec": train_time,
    }
    loss_path = os.path.join(run_dir, "loss.json")
    save_config(loss_path, loss_log)
    print(f"  Saved {loss_path} (train_time_sec={train_time:.1f})")

    config_path = os.path.join(run_dir, "config.json")
    save_config(config_path, extra_config)
    print(f"  Saved {config_path}")


def load_dataset_seeded(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    split: float,
    shuffle: bool,
    seed: int,
    device: torch.device,
):
    train_set_size = int(len(dataset) * split)
    valid_set_size = len(dataset) - train_set_size
    # random_split requires a CPU generator (only used for indexing)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    train_set, valid_set = torch.utils.data.random_split(
        dataset, [train_set_size, valid_set_size], generator=generator
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle, drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=shuffle, drop_last=True
    )
    return train_loader, valid_loader


def main(args: argparse.Namespace) -> None:
    # check for compatible device
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
        print("cuda not available, will use cpu")

    print(f"Using device: {args.device}")

    # convert dtype string to torch dtype
    args.dtype = torch.float32 if args.dtype == "float32" else torch.float64

    set_global_seed(args.seed)

    if args.train_dir is None:
        args.train_dir = os.path.join(
            "output", "benchmarking_" + time.strftime("%Y%m%d-%H%M%S")
        )
    os.makedirs(args.train_dir, exist_ok=True)

    print(f"Train output directory: {args.train_dir}")
    root_args = {k: v for k, v in vars(args).items()}
    args_path = os.path.join(args.train_dir, "args.json")
    save_config(args_path, root_args)
    print(f"  Saved {args_path}")

    # 6 configs × 2 sizes each; 4 FDNs per (config, size)
    worklist = [
        ("baseline", [8, 16, 32]),                    # 1: baseline, train I/O only
        ("hadamard", [8, 16, 32]),                    # 2: hadamard, train I/O only
        ("orthogonal", [8, 16, 32]),                  # 3: trained orthogonal
        ("householder", [8, 16, 32]),                 # 4: trained single Householder
        ("multi_householder", [8, 16, 32]),           # 5: trained multi-reflection Householder
        # ("scattering", [6]),                        # 6: scattering, each mix trained orthogonal
        # ("scattering_hadamard", [4, 8]),            # 7: scattering, last mix trained orthogonal, rest Hadamard
        # ("scattering_householder", [4, 6, 8]),      # 8: scattering, each mix trained Householder
    ]

    unique_sizes = sorted({n for _, n_list in worklist for n in n_list})
    delays_by_size: dict[int, list[torch.Tensor]] = {}
    for N in unique_sizes:
        delays_by_size[N] = [
            sample_coprime_delays(N, args.delay_min, args.delay_max)
            for _ in range(args.sets_per_config)
        ]

    for feedback_kind, n_list in worklist:
        for N in n_list:
            delay_low = args.delay_min
            delay_high = args.delay_max

            for delay_idx in range(args.sets_per_config):
                delay_lengths = delays_by_size[N][delay_idx]
                delay_list = delay_lengths.tolist()

                dataset_expand = (
                    args.num_scattering
                    if feedback_kind
                    in ("scattering", "scattering_hadamard", "scattering_householder")
                    else args.num
                )
                if dataset_expand is None:
                    dataset_expand = max(1, (args.nfft // 2 + 1) // 2000)

                for init_idx in range(args.inits_per_delay):
                    model, metadata = build_fdn(args, N, delay_lengths, feedback_kind)

                    run_dir = os.path.join(
                        args.train_dir,
                        feedback_kind,
                        f"N{N}",
                        f"set_{delay_idx:02d}_{init_idx:02d}",
                    )
                    print(
                        f"\nTraining job: type={feedback_kind}, N={N}, "
                        f"delay_set={delay_idx}, init={init_idx}"
                    )
                    print(f"  delay_lengths (samples): {delay_list}")
                    print(f"  -> {run_dir}")

                    config = {
                        "feedback_kind": feedback_kind,
                        "N": N,
                        "delay_index": delay_idx,
                        "init_index": init_idx,
                        "delay_lengths_samples": delay_lengths,
                        "delay_range": [delay_low, delay_high],
                        "samplerate": args.samplerate,
                        "nfft": args.nfft,
                        "max_epochs": args.max_epochs,
                        "patience": args.patience,
                        "patience_delta": args.patience_delta,
                        "lr": args.lr,
                        "batch_size": args.batch_size,
                        "dataset_expand": dataset_expand,
                        "mask_samples": args.mask_samples,
                        "sparsity_weight": args.sparsity_weight,
                        "alias_decay_db": args.alias_decay_db,
                        "num_reflections": args.num_reflections,
                        "seed": args.seed,
                        **metadata,
                    }

                    run_training(args, run_dir, model, dataset_expand, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--nfft", type=int, default=2**17, help="FFT size")
    parser.add_argument("--samplerate", type=int, default=48000, help="sampling rate")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float64",
        choices=["float32", "float64"],
        help="data type for tensors",
    )
    parser.add_argument("--num", type=int, default=2**8, help="dataset size")
    parser.add_argument(
        "--num_scattering",
        type=int,
        default=None,
        help="dataset size for scattering runs (defaults to nfft//2+1 // 2000)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed")
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to use for computation"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--max_epochs", type=int, default=1000, help="maximum number of epochs"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="early stopping patience in epochs",
    )
    parser.add_argument(
        "--patience_delta",
        type=float,
        default=0.001,
        help="minimum validation loss improvement to count as improvement",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--train_dir", type=str, help="directory to save training results"
    )
    parser.add_argument(
        "--sets_per_config",
        type=int,
        default=4,
        help="number of random delay sets per configuration",
    )
    parser.add_argument(
        "--inits_per_delay",
        type=int,
        default=4,
        help="number of independently-initialized FDN training runs per delay set",
    )
    parser.add_argument(
        "--num_reflections",
        type=int,
        default=None,
        help="number of Householder reflections for multi_householder (defaults to N)",
    )
    parser.add_argument(
        "--delay_min",
        type=int,
        default=500,
        help="minimum delay length (samples) for standard runs",
    )
    parser.add_argument(
        "--delay_max",
        type=int,
        default=3000,
        help="maximum delay length (samples) for standard runs",
    )

    parser.add_argument(
        "--mask_samples",
        type=int,
        default=2048,
        help="number of bins used for masked MSE loss",
    )
    parser.add_argument(
        "--sparsity_weight",
        type=float,
        default=0.2,
        help="weight for sparsity loss",
    )
    parser.add_argument(
        "--alias_decay_db",
        type=float,
        default=30.0,
        help="alias decay in dB",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="train/valid split",
    )

    main(parser.parse_args())
