#!/usr/bin/env python3
"""
Compute the average elapsed time for a given activity recorded in timings_single.csv.

Usage:
    python avg_time.py "Metric computation"
"""

import csv
import argparse
import statistics
import sys
from pathlib import Path


def average_time(csv_path: Path, activity_query: str) -> float | None:
    """Return mean of `time` where `activity` matches activity_query, ignoring case & whitespace."""
    times = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)

        # Detect the column name that holds activity (fallback if it's not exactly 'activity')
        header_lower = {h.lower(): h for h in reader.fieldnames or []}
        activity_col = header_lower.get("activity")
        if activity_col is None:
            sys.exit(
                f'Could not find "activity" column. Columns present: {", ".join(reader.fieldnames or [])}'
            )

        for row in reader:
            raw = row.get(activity_col, "")
            if raw.strip().lower() == activity_query.strip().lower():
                try:
                    times.append(float(row["time"]))
                except (ValueError, KeyError):
                    continue
    return statistics.mean(times) if times else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Average timing for a specific activity.")
    parser.add_argument("activity", help="Activity string to filter on (case-insensitive)")
    parser.add_argument("--csv", default="timings_single.csv", help="CSV file path")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        sys.exit(f"File not found: {csv_path}")

    mean_time = average_time(csv_path, args.activity)
    if mean_time is None:
        sys.exit(f'No rows found for activity "{args.activity}".')
    print(f'Average time for "{args.activity}": {mean_time:.3f} seconds')


if __name__ == "__main__":
    main()
