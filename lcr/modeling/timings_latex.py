#!/usr/bin/env python3
"""
Read timings_single.csv and generate a LaTeX table of average times per activity.

Usage:
    python timings_to_latex.py               # assumes timings_single.csv in cwd
    python timings_to_latex.py --csv timings_single.csv \
           --caption "Timings for RF & CNN models." --label tab:ModelTiming
"""

import csv
import statistics
import argparse
import re
from pathlib import Path
from typing import Optional, Dict, List

# ── helpers ────────────────────────────────────────────────────────────────────
_TDELTA_RE = re.compile(
    r"""^\s*(?:(?P<days>\d+)\s+days?\s+)?  # optional 'X days'
        (?P<hours>\d{1,2}):
        (?P<mins>\d{2}):
        (?P<secs>\d{2}(?:\.\d+)?)\s*$      # seconds[.micro]
    """,
    re.VERBOSE,
)

def to_seconds(cell: str) -> Optional[float]:
    """Return `cell` expressed in seconds (float) or None if it can't be parsed."""
    # numeric already?
    try:
        return float(cell)
    except ValueError:
        pass

    # 'X days HH:MM:SS.micro'
    m = _TDELTA_RE.match(cell)
    if not m:
        return None
    days  = int(m.group("days") or 0)
    hours = int(m.group("hours"))
    mins  = int(m.group("mins"))
    secs  = float(m.group("secs"))
    return days * 86_400 + hours * 3_600 + mins * 60 + secs

# ── core ───────────────────────────────────────────────────────────────────────
def mean_times(csv_path: Path) -> Dict[str, float]:
    """Return {activity: mean_seconds} for every activity in csv_path."""
    buckets: Dict[str, List[float]] = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        # find 'activity' & 'time' columns case-insensitively
        headers = {h.lower(): h for h in reader.fieldnames or []}
        act_col = headers.get("activity")
        time_col = headers.get("time")
        if act_col is None or time_col is None:
            raise ValueError(
                f"Expected 'activity' and 'time' columns, found: {', '.join(reader.fieldnames or [])}"
            )

        for row in reader:
            activity = row[act_col].strip()
            sec = to_seconds(row[time_col])
            if sec is not None:
                buckets.setdefault(activity, []).append(sec)

    return {a: statistics.mean(t) for a, t in buckets.items() if t}

def latex_table(means: Dict[str, float], caption: str, label: str) -> str:
    """Return LaTeX code for the mean-time table (sorted alphabetically)."""
    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        r"    \footnotesize",
        r"    \begin{tabular}{lc}",
        r"        \toprule",
        r"        \textbf{Feature} & \textbf{Mean Compute Time (s)} \\",
        r"        \midrule",
    ]
    for activity, avg in sorted(means.items()):
        lines.append(f"        {activity} & {avg:.3f} \\\\")
    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)

# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate LaTeX table from timings_single.csv")
    p.add_argument("--csv", default="timings_single.csv", help="Path to CSV file")
    p.add_argument("--caption", default="Timings for training and testing RF and CNN models.",
                   help="LaTeX caption")
    p.add_argument("--label", default="tab:Modeltiming", help="LaTeX label")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        p.error(f"CSV file not found: {csv_path}")

    mean_dict = mean_times(csv_path)
    if not mean_dict:
        p.error("No valid timing data found in CSV.")

    print(latex_table(mean_dict, args.caption, args.label))
