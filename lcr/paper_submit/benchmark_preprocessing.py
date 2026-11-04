#!/usr/bin/env python3
"""Benchmark RF features, quality metrics, and label construction."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import xarray as xr

from feature_generation.generate_features import (
    METRIC_CALCULATIONS,
    VECTORIZED_FEATURES,
    flatten_variables,
    scalar_value,
)
from precomputed_labeling import (
    DEFAULT_METRICS_INFO,
    generate_classification_labels,
)


RAW_FIELDS = (
    "variable",
    "stage",
    "operation",
    "collection",
    "compression",
    "timestep",
    "seconds",
)
SUMMARY_FIELDS = (
    "variable",
    "stage",
    "operation",
    "count",
    "mean_seconds",
    "std_seconds",
    "min_seconds",
    "max_seconds",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute and time RF features, compression quality metrics, and "
            "label construction on a small leading timestep subset."
        )
    )
    parser.add_argument("--config", required=True, help="Model JSON configuration")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timesteps", type=int, default=10)
    parser.add_argument(
        "--variable",
        action="append",
        dest="variables",
        help="Benchmark only this variable; may be supplied more than once",
    )
    return parser.parse_args()


def source_paths(config, variable, subdirectory, prefix, suffix):
    filename = f"{prefix}{variable}{suffix}"
    original = (
        Path(config["OrigPath"]) / subdirectory / "orig" / filename
    )
    compressed = {
        compression: (
            Path(config["CompPath"])
            / subdirectory
            / compression
            / filename
        )
        for compression in config["CompDirs"]
    }
    return original, compressed


def load_variable(path, variable, timesteps):
    if not path.is_file():
        raise FileNotFoundError(path)
    with xr.open_dataset(path) as dataset:
        if variable not in dataset:
            raise KeyError(f"{variable!r} is not present in {path}")
        data = dataset[variable]
        if data.sizes.get("time", 0) < timesteps:
            raise ValueError(
                f"{path} has fewer than {timesteps} timesteps"
            )
        return data.isel(time=slice(0, timesteps)).load()


def add_record(
    records,
    variable,
    stage,
    operation,
    collection,
    compression,
    timestep,
    seconds,
):
    records.append(
        {
            "variable": variable,
            "stage": stage,
            "operation": operation,
            "collection": collection,
            "compression": compression,
            "timestep": timestep,
            "seconds": float(seconds),
        }
    )


def benchmark_features(
    ldcpy,
    records,
    variable,
    collection,
    original,
    features,
    timesteps,
):
    for timestep in range(timesteps):
        calculator = ldcpy.Datasetcalcs(
            original.isel(time=timestep),
            "cam-fv",
            ["lat", "lon"],
            weighted=False,
        )
        for feature in features:
            started = time.perf_counter()
            if feature in VECTORIZED_FEATURES:
                value = calculator.get_calc(feature)
            else:
                value = calculator.get_single_calc(feature)
            scalar_value(value)
            elapsed = time.perf_counter() - started
            add_record(
                records,
                variable,
                "feature",
                feature,
                collection,
                "",
                timestep + 1,
                elapsed,
            )


def benchmark_metrics(
    ldcpy,
    records,
    variable,
    collection,
    original,
    compressed_by_level,
    metrics,
    timesteps,
):
    metric_values = {
        metric: {
            compression: np.empty(timesteps, dtype=np.float64)
            for compression in compressed_by_level
        }
        for metric in metrics
    }
    for compression, compressed in compressed_by_level.items():
        original_aligned, compressed_aligned = xr.align(
            original, compressed, join="inner"
        )
        if original_aligned.sizes.get("time", 0) < timesteps:
            raise ValueError(
                f"Only {original_aligned.sizes.get('time', 0)} aligned "
                f"timesteps are available for {variable} {compression}"
            )
        for timestep in range(timesteps):
            for metric in metrics:
                started = time.perf_counter()
                calculator = ldcpy.Diffcalcs(
                    original_aligned.isel(time=timestep),
                    compressed_aligned.isel(time=timestep),
                    data_type="cam-fv",
                    aggregate_dims=["lat", "lon"],
                )
                value = scalar_value(
                    calculator.get_diff_calc(METRIC_CALCULATIONS[metric])
                )
                elapsed = time.perf_counter() - started
                metric_values[metric][compression][timestep] = value
                add_record(
                    records,
                    variable,
                    "metric",
                    metric,
                    collection,
                    compression,
                    timestep + 1,
                    elapsed,
                )
    return metric_values


def benchmark_labels(
    records,
    variable,
    collection,
    metric_values,
    metrics,
    compression_levels,
    timesteps,
):
    rules = {metric: DEFAULT_METRICS_INFO[metric] for metric in metrics}
    operation = "+".join(metrics)
    for timestep in range(timesteps):
        one_timestep = {
            metric: {
                compression: np.asarray([values[timestep]])
                for compression, values in metric_values[metric].items()
            }
            for metric in metrics
        }
        started = time.perf_counter()
        generate_classification_labels(
            rules,
            one_timestep,
            compression_levels,
        )
        elapsed = time.perf_counter() - started
        add_record(
            records,
            variable,
            "label",
            operation,
            collection,
            "",
            timestep + 1,
            elapsed,
        )


def summary_rows(records, include_variable):
    grouped = {}
    for record in records:
        key = (
            record["variable"] if include_variable else "ALL",
            record["stage"],
            record["operation"],
        )
        grouped.setdefault(key, []).append(record["seconds"])

    rows = []
    for key in sorted(grouped):
        values = grouped[key]
        rows.append(
            {
                "variable": key[0],
                "stage": key[1],
                "operation": key[2],
                "count": len(values),
                "mean_seconds": statistics.fmean(values),
                "std_seconds": statistics.pstdev(values),
                "min_seconds": min(values),
                "max_seconds": max(values),
            }
        )
    return rows


def write_table(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for field in (
                "seconds",
                "mean_seconds",
                "std_seconds",
                "min_seconds",
                "max_seconds",
            ):
                if field in formatted:
                    formatted[field] = f"{formatted[field]:.9f}"
            writer.writerow(formatted)


def main():
    args = parse_args()
    if args.timesteps < 1:
        raise ValueError("--timesteps must be positive")

    config_path = Path(args.config).resolve()
    with config_path.open() as stream:
        config = json.load(stream)

    variables = flatten_variables(config["VarList"])
    if args.variables:
        missing = set(args.variables) - set(variables)
        if missing:
            raise ValueError(
                f"Variables not present in the configuration: {sorted(missing)}"
            )
        variables = args.variables

    metrics = list(config["Metric"])
    unsupported = set(metrics) - set(METRIC_CALCULATIONS)
    if unsupported:
        raise ValueError(f"Unsupported metrics: {sorted(unsupported)}")
    missing_rules = set(metrics) - set(DEFAULT_METRICS_INFO)
    if missing_rules:
        raise ValueError(f"No label rules for: {sorted(missing_rules)}")

    ldcpy_path = config.get("OptLdcpyDevPath")
    if ldcpy_path and ldcpy_path not in sys.path:
        sys.path.insert(0, ldcpy_path)
    import ldcpy

    subdirectories = config["SubDirs"]
    prefixes = config["FilenamePre"]
    suffixes = config["FilenamePost"]
    if not (len(subdirectories) == len(prefixes) == len(suffixes)):
        raise ValueError(
            "SubDirs, FilenamePre, and FilenamePost must have equal lengths"
        )

    records = []
    for variable_index, variable in enumerate(variables, start=1):
        print(f"[{variable_index}/{len(variables)}] {variable}", flush=True)
        for subdirectory, prefix, suffix in zip(
            subdirectories, prefixes, suffixes
        ):
            original_path, compressed_paths = source_paths(
                config, variable, subdirectory, prefix, suffix
            )
            original = load_variable(original_path, variable, args.timesteps)
            compressed = {
                level: load_variable(path, variable, args.timesteps)
                for level, path in compressed_paths.items()
            }
            collection = f"{subdirectory}_orig"
            benchmark_features(
                ldcpy,
                records,
                variable,
                collection,
                original,
                config["RFFeatureList"],
                args.timesteps,
            )
            metric_values = benchmark_metrics(
                ldcpy,
                records,
                variable,
                collection,
                original,
                compressed,
                metrics,
                args.timesteps,
            )
            benchmark_labels(
                records,
                variable,
                collection,
                metric_values,
                metrics,
                config["CompDirs"],
                args.timesteps,
            )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "preprocessing_timing_samples.txt"
    variable_path = output_dir / "preprocessing_timing_by_variable.txt"
    overall_path = output_dir / "preprocessing_timing_overall.txt"
    notes_path = output_dir / "preprocessing_timing_notes.txt"
    write_table(raw_path, RAW_FIELDS, records)
    write_table(
        variable_path,
        SUMMARY_FIELDS,
        summary_rows(records, include_variable=True),
    )
    write_table(
        overall_path,
        SUMMARY_FIELDS,
        summary_rows(records, include_variable=False),
    )
    notes_path.write_text(
        "\n".join(
            (
                f"config={config_path}",
                f"variables={len(variables)}",
                f"timesteps={args.timesteps}",
                f"features={','.join(config['RFFeatureList'])}",
                f"metrics={','.join(metrics)}",
                f"compressions={','.join(config['CompDirs'])}",
                "timing_clock=time.perf_counter",
                "std_definition=population standard deviation",
                "feature_timer=ldcpy feature calculation call; input file loading and calculator construction excluded",
                "metric_timer=Diffcalcs construction plus one metric calculation; input file loading excluded",
                "label_timer=classification-label construction from already computed metric values",
                "timestep_subset=first N positional timesteps from each source file",
            )
        )
        + "\n"
    )
    print(f"Raw timings: {raw_path}")
    print(f"Per-variable summary: {variable_path}")
    print(f"Overall summary: {overall_path}")


if __name__ == "__main__":
    main()
