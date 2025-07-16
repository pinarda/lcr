#!/usr/bin/env python3
"""
print_array_info.py

Report shape, element count, dtype, and memory use for NumPy .npy or .npz files.

Usage
-----
# Single .npy file
python print_array_info.py path/to/array.npy

# .npz file (lists all arrays inside)
python print_array_info.py path/to/archive.npz

# .npz file, report just one stored array by key
python print_array_info.py path/to/archive.npz --key my_array

# Print only the total element count (quiet)
python print_array_info.py path/to/array.npy --count

# Human-readable memory is default; to suppress it:
python print_array_info.py path/to/array.npy --no-human
"""
import argparse
import sys
import numpy as np
from pathlib import Path


def human_bytes(num_bytes: int) -> str:
    """Return a human-readable byte string."""
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    x = float(num_bytes)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:0.2f}{u}"
        x /= 1024.0
    return f"{num_bytes}B"  # fallback


def summarize_array(name: str, arr: np.ndarray, show_human: bool = True):
    """Print array summary."""
    elems = arr.size
    nbytes = arr.nbytes
    hb = human_bytes(nbytes) if show_human else None
    shape_str = "x".join(str(s) for s in arr.shape)
    print(f"[{name}] shape={shape_str} dtype={arr.dtype} size={elems} nbytes={nbytes}"
          + (f" ({hb})" if hb else ""))


def load_npy(path: Path) -> np.ndarray:
    try:
        return np.load(path, allow_pickle=False)
    except Exception as e:
        print(f"ERROR loading {path}: {e}", file=sys.stderr)
        sys.exit(1)


def load_npz(path: Path):
    try:
        return np.load(path, allow_pickle=False)
    except Exception as e:
        print(f"ERROR loading {path}: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Print size/shape info for NumPy arrays.")
    ap.add_argument("path", help=".npy or .npz file path")
    ap.add_argument("--key", help="For .npz: name of array to report (else all).")
    ap.add_argument("--count", action="store_true",
                    help="Print only the element count (if a single array is selected).")
    ap.add_argument("--no-human", action="store_true",
                    help="Suppress human-readable byte output.")
    args = ap.parse_args()

    p = Path(args.path)
    if not p.is_file():
        print(f"ERROR: {p} not found.", file=sys.stderr)
        return 2

    suffix = p.suffix.lower()

    if suffix == ".npy":
        arr = load_npy(p)
        if args.count:
            print(arr.size)
            return 0
        summarize_array(p.name, arr, show_human=not args.no_human)
        return 0

    if suffix == ".npz":
        data = load_npz(p)  # returns NpzFile (dict-like)

        # Pick which arrays to show
        if args.key:
            if args.key not in data.files:
                print(f"ERROR: key '{args.key}' not found in {p.name}. Available: {data.files}",
                      file=sys.stderr)
                return 3
            arr = data[args.key]
            if args.count:
                print(arr.size)
                return 0
            summarize_array(args.key, arr, show_human=not args.no_human)
            return 0

        # No key specified: show all arrays
        total_elems = 0
        total_bytes = 0
        for k in data.files:
            arr = data[k]
            summarize_array(k, arr, show_human=not args.no_human)
            total_elems += arr.size
            total_bytes += arr.nbytes

        print("-" * 60)
        hb = human_bytes(total_bytes) if not args.no_human else None
        print(f"TOTAL: arrays={len(data.files)} elems={total_elems} bytes={total_bytes}"
              + (f" ({hb})" if hb else ""))
        return 0

    print(f"ERROR: unsupported extension '{suffix}'. Use .npy or .npz.", file=sys.stderr)
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
