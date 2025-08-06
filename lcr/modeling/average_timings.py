#!/usr/bin/env python3
"""
Compute the average elapsed time for a given activity recorded in timings_single.csv.

Works with either:
    12.345
or
    0 days 00:00:12.345678
"""

import csv
import argparse
import statistics
import sys
import re
from pathlib import Path
from typing import Optional

# ── helper ──────────────────────────────────────────────────────────────────────
_TIMEDelta_RE = re.compile(
    r"""
    ^\s*
    (?:(?P<days>\d+)\s+days?\s+)?
    (?P<hours>\d{1,2}):
    (?P<mins>\d{2}):
    (?P<secs>\d{2}(?:\.\d+)?)
    \s*$
    """,
    re.VERBOSE,
)

def parse_seconds(cell: str) -> Optional[float]:
    """
    Convert a CSV cell to seconds (float).
    Accepts either a bare number or a string like '2 days 01:23:45.6'.
    Returns None if parsing fails.
    """
    # 1) bare number
    try:
        return float(cell)
    except ValueError:
        pass

    # 2) 'X days HH:MM:SS.micro'
    m = _TIMEDelta_RE.match(cell)
    if not m:
        return None

    days  = int(m.group("days") or 0)
    hours = int(m.group("hours"))
    mins  = int(m.group("mins"))
    secs  = float(m.group("secs"))
    return days * 86_400 + hours * 3_600 + mins * 60 + secs

# ── average routine ─────────────────────────────────────────────────────────────
def average_time(csv_path: Path, activity_query: str) -> Optional[float]:
    times = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)

        # find the 'activity' column, case-insensitive
        cols = {h.lower(): h for h in reader.fieldnames or []}
        activity_col = cols.get("activity")
        if activity_col is None:
            sys.exit(f'"activity" column not found. Columns: {", ".join(reader.fieldnames or [])}')

        for row in reader:
            if row.get(activity_col, "").strip().lower() == activity_query.strip().lower():
                sec = parse_seconds(row["time"])
                if sec is not None:
                    times.append(sec)

    return statistics.mean(times) if times else None

# ── CLI ─────────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="Average timing for a specific activity.")
    p.add_argument("activity", help="Activity string (case-insensitive match)")
    p.add_argument("--csv", default="timings_single.csv", help="CSV file path")
    args = p.parse_args()

    path = Path(args.csv)
    if not path.is_file():
        sys.exit(f"File not found: {path}")

    mean = average_time(path, args.activity)
    if mean is None:
        sys.exit(f'No valid rows for activity "{args.activity}".')

    print(f'Average time for "{args.activity}": {mean:.3f} seconds')

if __name__ == "__main__":
    main()
