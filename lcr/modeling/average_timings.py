#!/usr/bin/env python3
"""
Compute the average elapsed time for a given activity recorded in timings_single.csv.

Usage:
    python avg_time.py "Random Forest and Decision Tree training"
"""

import csv
import argparse
import statistics
import sys
from pathlib import Path

def average_time(csv_path: Path, activity: str):
    """Return the mean of the `time` column for rows whose `activity` matches (case-sensitive)."""
    times = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("activity") == activity:
                # convert to float; skip rows with invalid numbers
                try:
                    times.append(float(row["time"]))
                except (ValueError, KeyError):
                    continue
    return statistics.mean(times) if times else None

def main() -> None:
    parser = argparse.ArgumentParser(description="Average timing for a specific activity.")
    parser.add_argument("activity", help="Exact activity string to filter on")
    parser.add_argument("--csv", default="timings_single.csv", help="Path to CSV file (default: timings_single.csv)")
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
