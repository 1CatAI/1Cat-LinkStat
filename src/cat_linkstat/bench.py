from __future__ import annotations

import argparse
import importlib.resources as resources
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__

BENCH_BINARY_STEM = "1catlinkstat-nvlink-bench"
BENCH_SOURCE_NAME = "nvlink_microbench.cu"


def default_cache_dir(environ: dict[str, str] | None = None, home: Path | None = None) -> Path:
    env = os.environ if environ is None else environ
    if home is None:
        home = Path.home()
    cache_root = env.get("XDG_CACHE_HOME")
    if cache_root:
        return Path(cache_root) / "1catlinkstat"
    return home / ".cache" / "1catlinkstat"


def binary_path(cache_dir: Path) -> Path:
    return cache_dir / f"{BENCH_BINARY_STEM}-{__version__}"


def needs_rebuild(source: Path, binary: Path) -> bool:
    if not binary.exists():
        return True
    return source.stat().st_mtime_ns > binary.stat().st_mtime_ns


def build_command(nvcc: str, source: Path, binary: Path) -> list[str]:
    return [
        nvcc,
        "-O3",
        "-std=c++17",
        str(source),
        "-o",
        str(binary),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="1catlinkstat-bench",
        description="Build and run the bundled 1CatLinkStat NVLink microbenchmark.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory used to store the compiled benchmark binary.",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only build the bundled CUDA benchmark and print the binary path.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuilding the bundled CUDA benchmark even if a cached binary exists.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the nvcc command while compiling the benchmark.",
    )
    return parser


def compile_benchmark(source: Path, cache_dir: Path, force: bool, verbose: bool) -> Path:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError(
            "nvcc not found. Install NVIDIA CUDA Toolkit or nvidia-cuda-toolkit before running 1catlinkstat-bench."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    binary = binary_path(cache_dir)
    if not force and not needs_rebuild(source, binary):
        return binary

    temp_binary = binary.with_suffix(".tmp")
    if temp_binary.exists():
        temp_binary.unlink()

    command = build_command(nvcc, source, temp_binary)
    if verbose:
        print("Compiling bundled CUDA microbenchmark:")
        print(" ".join(command), file=sys.stderr)

    completed = subprocess.run(
        command,
        capture_output=not verbose,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        message = ["Failed to compile bundled CUDA microbenchmark."]
        if completed.stdout:
            message.append(completed.stdout.strip())
        if completed.stderr:
            message.append(completed.stderr.strip())
        raise RuntimeError("\n".join(part for part in message if part))

    temp_binary.replace(binary)
    return binary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, bench_args = parser.parse_known_args(argv)
    cache_dir = args.cache_dir or default_cache_dir()

    try:
        source_ref = resources.files("cat_linkstat").joinpath(BENCH_SOURCE_NAME)
        with resources.as_file(source_ref) as source:
            binary = compile_benchmark(source, cache_dir, force=args.rebuild, verbose=args.verbose)
    except RuntimeError as exc:
        print(f"1CatLinkStat-bench: {exc}", file=sys.stderr)
        return 1

    if args.build_only:
        print(binary)
        return 0

    completed = subprocess.run([str(binary), *bench_args], check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
