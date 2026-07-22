#!/usr/bin/env python3
"""Benchmark the official pi05 base checkpoint with the upstream Aloha policy config."""

import argparse
import json
import pathlib
import platform
import time

import jax
import numpy as np

from openpi.policies import aloha_policy
from openpi.policies import policy_config
from openpi.training import config as _config

OFFICIAL_CONFIG_NAME = "pi05_aloha"
DEFAULT_CHECKPOINT = "/home/user/.cache/openpi/checkpoints/pi05_base/params"


def _checkpoint_root(path: str) -> pathlib.Path:
    checkpoint = pathlib.Path(path).expanduser().resolve()
    if checkpoint.name == "params" and (checkpoint / "_METADATA").is_file():
        checkpoint = checkpoint.parent
    if not (checkpoint / "params").is_dir():
        raise FileNotFoundError(f"Checkpoint must contain params/: {checkpoint}")
    if not (checkpoint / "assets" / "trossen" / "norm_stats.json").is_file():
        raise FileNotFoundError(f"Checkpoint has no official trossen norm stats: {checkpoint}")
    return checkpoint


def _stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure official pi05_base inference frequency using the upstream pi05_aloha config."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--warmup-calls", type=int, default=3)
    parser.add_argument("--frequency-calls", type=int, default=100)
    parser.add_argument(
        "--long-run-seconds",
        type=float,
        default=0.0,
        help="Run for this many seconds after warmup. A positive value overrides --frequency-calls.",
    )
    parser.add_argument("--prompt", default="do something")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--output-dir", default="offline_eval/pi05_base_frequency")
    args = parser.parse_args()
    if args.num_steps <= 0 or args.frequency_calls <= 0 or args.progress_every <= 0:
        parser.error("--num-steps, --frequency-calls, and --progress-every must be positive")
    if args.warmup_calls < 0 or args.long_run_seconds < 0:
        parser.error("--warmup-calls and --long-run-seconds must be non-negative")
    return args


def main() -> None:
    args = _parse_args()
    checkpoint = _checkpoint_root(args.checkpoint)
    train_config = _config.get_config(OFFICIAL_CONFIG_NAME)
    policy = policy_config.create_trained_policy(
        train_config,
        checkpoint,
        default_prompt=args.prompt,
        sample_kwargs={"num_steps": args.num_steps},
    )

    # This is the input generator shipped with the official Aloha policy adapter.
    observation = aloha_policy.make_aloha_example()
    observation["prompt"] = args.prompt

    print(f"config           = {OFFICIAL_CONFIG_NAME}")
    print(f"checkpoint       = {checkpoint}")
    print("input            = official synthetic Aloha example (no dataset I/O)")
    print(f"jax devices      = {jax.devices()}")
    print(f"num steps        = {args.num_steps}")
    print(f"warmup calls     = {args.warmup_calls}")
    if args.long_run_seconds > 0:
        print(f"long run seconds = {args.long_run_seconds:.1f}")
    else:
        print(f"frequency calls  = {args.frequency_calls}")

    for index in range(args.warmup_calls):
        policy.infer(observation)
        print(f"  warmup {index + 1}/{args.warmup_calls}")

    wall_ms_values: list[float] = []
    model_ms_values: list[float] = []
    run_started = time.perf_counter()
    while True:
        if args.long_run_seconds > 0:
            if wall_ms_values and time.perf_counter() - run_started >= args.long_run_seconds:
                break
        elif len(wall_ms_values) >= args.frequency_calls:
            break

        call_started = time.perf_counter()
        result = policy.infer(observation)
        wall_ms_values.append((time.perf_counter() - call_started) * 1000.0)
        model_ms_values.append(float(result["policy_timing"]["infer_ms"]))

        if len(wall_ms_values) % args.progress_every == 0:
            elapsed = time.perf_counter() - run_started
            print(f"  inferred {len(wall_ms_values)} calls in {elapsed:.1f}s ({len(wall_ms_values) / elapsed:.2f} Hz)")

    elapsed_seconds = time.perf_counter() - run_started
    wall_hz_values = [1000.0 / value for value in wall_ms_values]
    model_hz_values = [1000.0 / value for value in model_ms_values]
    summary = {
        "mode": "official_pi05_base_frequency",
        "config": OFFICIAL_CONFIG_NAME,
        "checkpoint": str(checkpoint),
        "input": "aloha_policy.make_aloha_example",
        "includes_dataset_io": False,
        "prompt": args.prompt,
        "num_steps": args.num_steps,
        "warmup_calls": args.warmup_calls,
        "requested_long_run_seconds": args.long_run_seconds,
        "elapsed_seconds": elapsed_seconds,
        "frequency_calls": len(wall_ms_values),
        "aggregate_hz": len(wall_ms_values) / elapsed_seconds,
        "wall_ms": _stats(wall_ms_values),
        "wall_hz": _stats(wall_hz_values),
        "model_ms": _stats(model_ms_values),
        "model_hz": _stats(model_hz_values),
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jax_devices": [str(device) for device in jax.devices()],
        },
    }

    output_dir = pathlib.Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pi05_base_frequency.json"
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")

    print(f"aggregate Hz     = {summary['aggregate_hz']:.3f}")
    print(f"wall latency ms  = {summary['wall_ms']}")
    print(f"model latency ms = {summary['model_ms']}")
    print(f"saved            = {output_path.resolve()}")


if __name__ == "__main__":
    main()
