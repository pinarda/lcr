#!/usr/bin/env python3
"""Generate reusable 2,000-timestep RF features and quality metrics."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import json
import logging
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import xarray as xr


VECTORIZED_FEATURES = {
    "ns_con_var",
    "ew_con_var",
    "w_e_first_differences",
    "n_s_first_differences",
    "fftratio",
    "fftmax",
    "w_e_first_differences_max",
    "n_s_first_differences_max",
    "mean",
    "magnitude_range",
}

METRIC_CALCULATIONS = {
    "dssim": "ssim_fp",
    "pcc": "pearson_correlation_coefficient",
    "spre": "spatial_rel_error",
    "ks": "ks_p_value",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute RF input features and compression quality metrics."
    )
    parser.add_argument("--config", required=True, help="Model JSON configuration")
    parser.add_argument(
        "--quality-config",
        action="append",
        default=[],
        help=(
            "Configuration whose CompDirs/Metric combinations should be computed; "
            "may be supplied more than once. Defaults to --config."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="./data",
        help="Directory for intermediate feature files",
    )
    parser.add_argument(
        "--metrics-dir",
        help="Directory for intermediate quality-metric files; defaults to --output-dir",
    )
    parser.add_argument("--timesteps", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--mode",
        choices=("both", "features", "metrics", "combine"),
        default="both",
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--variable", help="Compute one named variable")
    selector.add_argument(
        "--variable-index",
        type=int,
        help="Compute one variable by one-based position in VarList",
    )
    parser.add_argument(
        "--compression-index",
        type=int,
        help="Compute one compression level by one-based numeric order",
    )
    parser.add_argument(
        "--combined-output-dir",
        help="Directory for final per-variable NetCDF files",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing output files"
    )
    return parser.parse_args()


def flatten_variables(value):
    variables = []
    for item in value:
        if isinstance(item, list):
            variables.extend(flatten_variables(item))
        else:
            variables.append(item)
    return variables


def select_variables(config, variable, variable_index):
    variables = flatten_variables(config["VarList"])
    if len(variables) != len(set(variables)):
        raise ValueError("VarList contains duplicate variable names")
    if variable is not None:
        if variable not in variables:
            raise ValueError(f"Variable {variable!r} is not present in VarList")
        return [variable]
    if variable_index is not None:
        if not 1 <= variable_index <= len(variables):
            raise ValueError(
                f"Variable index must be between 1 and {len(variables)}, inclusive"
            )
        return [variables[variable_index - 1]]
    return variables


def compression_sort_key(name):
    match = re.search(r"(\d+)(?!.*\d)", name)
    if match:
        return (0, int(match.group(1)), name)
    return (1, 0, name)


def quality_requests(config_paths):
    requests = {}
    for config_path in config_paths:
        with Path(config_path).open() as stream:
            quality_config = json.load(stream)
        for compression_directory in quality_config["CompDirs"]:
            metrics = requests.setdefault(compression_directory, [])
            for metric in quality_config["Metric"]:
                if metric not in METRIC_CALCULATIONS:
                    raise ValueError(f"Unsupported quality metric {metric!r}")
                if metric not in metrics:
                    metrics.append(metric)
    return {
        compression: requests[compression]
        for compression in sorted(requests, key=compression_sort_key)
    }


def select_compression(requests, compression_index):
    if compression_index is None:
        return requests
    compressions = list(requests)
    if not 1 <= compression_index <= len(compressions):
        raise ValueError(
            f"Compression index must be between 1 and {len(compressions)}, inclusive"
        )
    selected = compressions[compression_index - 1]
    return {selected: requests[selected]}


def chunk_ranges(timesteps, workers):
    if timesteps < 1:
        raise ValueError("--timesteps must be positive")
    if workers < 1:
        raise ValueError("--workers must be positive")
    count = min(timesteps, workers)
    base, remainder = divmod(timesteps, count)
    ranges = []
    start = 0
    for index in range(count):
        stop = start + base + (1 if index < remainder else 0)
        ranges.append((start, stop))
        start = stop
    return ranges


def feature_path(output_dir, variable, collection_label, feature, timesteps):
    return output_dir / (
        f"{variable}_{collection_label}_FEATURE_{feature}_all_"
        f"time{timesteps}_second.nc"
    )


def metric_path(
    output_dir, variable, original_label, compressed_label, metric, timesteps
):
    return output_dir / (
        f"{variable}_{original_label}_{compressed_label}_{metric}_"
        f"time{timesteps}_second.npy"
    )


def scalar_value(result):
    values = np.asarray(result.values if hasattr(result, "values") else result)
    if values.size != 1:
        raise ValueError(
            f"Calculation returned {values.size} values; expected exactly one"
        )
    return values.reshape(-1)[0]


def normalized_values(result, expected, description):
    values = np.asarray(result.values if hasattr(result, "values") else result)
    values = values.reshape(-1)
    if values.size != expected:
        raise ValueError(
            f"{description} produced {values.size} values; expected {expected}"
        )
    return values


def write_feature(values, destination, variable, feature, source_path, timesteps):
    feature_data = xr.DataArray(
        values,
        dims=("timestep",),
        coords={"timestep": np.arange(1, timesteps + 1)},
        name=feature,
        attrs={
            "variable": variable,
            "feature": feature,
            "source_file": str(source_path),
            "timesteps": timesteps,
        },
    )
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        feature_data.to_netcdf(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_metric(values, destination):
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, values)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_dataset(dataset, destination):
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        dataset.to_netcdf(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def valid_existing_feature(path, timesteps):
    try:
        with xr.open_dataarray(path) as feature_data:
            return np.asarray(feature_data.values).size == timesteps
    except (OSError, ValueError):
        return False


def valid_existing_metric(path, timesteps):
    try:
        return np.asarray(np.load(path)).size == timesteps
    except (OSError, ValueError):
        return False


def verify_combined_outputs(
    feature_pathname,
    metric_pathname,
    expected_collections,
    expected_compressions,
    timesteps,
):
    expected_timestep = np.arange(1, timesteps + 1)
    with xr.open_dataset(feature_pathname) as dataset:
        if dataset["feature_values"].dims != (
            "collection",
            "timestep",
            "feature",
        ):
            raise ValueError(f"Unexpected feature dimension order in {feature_pathname}")
        if list(dataset["collection"].values.astype(str)) != expected_collections:
            raise ValueError(f"Unexpected collection order in {feature_pathname}")
        if not np.array_equal(dataset["timestep"].values, expected_timestep):
            raise ValueError(f"Unexpected timestep order in {feature_pathname}")
    with xr.open_dataset(metric_pathname) as dataset:
        if dataset["quality_metric_values"].dims != (
            "collection",
            "compression",
            "timestep",
            "metric",
        ):
            raise ValueError(f"Unexpected metric dimension order in {metric_pathname}")
        if list(dataset["compression"].values.astype(str)) != expected_compressions:
            raise ValueError(f"Unexpected compression order in {metric_pathname}")
        if not np.array_equal(dataset["timestep"].values, expected_timestep):
            raise ValueError(f"Unexpected timestep order in {metric_pathname}")


def _import_ldcpy(ldcpy_path):
    if ldcpy_path and ldcpy_path not in sys.path:
        sys.path.insert(0, ldcpy_path)
    import ldcpy

    return ldcpy


def feature_chunk_worker(task):
    source_specs, variable, start, stop, ldcpy_path = task
    ldcpy = _import_ldcpy(ldcpy_path)
    chunk_results = {}
    for collection_label, source_name, features in source_specs:
        with xr.open_dataset(source_name) as dataset:
            if variable not in dataset:
                raise KeyError(f"{variable!r} is not present in {source_name}")
            if "time" not in dataset[variable].dims:
                raise ValueError(f"{source_name} has no time dimension")
            data = dataset[variable].isel(time=slice(start, stop)).load()

        expected = stop - start
        values_by_feature = {}
        vector_features = [f for f in features if f in VECTORIZED_FEATURES]
        if vector_features:
            calculator = ldcpy.Datasetcalcs(
                data, "cam-fv", ["lat", "lon"], weighted=False
            )
            for feature in vector_features:
                values_by_feature[feature] = normalized_values(
                    calculator.get_calc(feature),
                    expected,
                    f"Feature {feature!r} for {variable!r}",
                )

        scalar_features = [f for f in features if f not in VECTORIZED_FEATURES]
        if scalar_features:
            scalar_results = {
                feature: np.empty(expected, dtype=np.float64)
                for feature in scalar_features
            }
            for local_index in range(expected):
                calculator = ldcpy.Datasetcalcs(
                    data.isel(time=local_index),
                    "cam-fv",
                    ["lat", "lon"],
                    weighted=False,
                )
                for feature in scalar_features:
                    scalar_results[feature][local_index] = scalar_value(
                        calculator.get_single_calc(feature)
                    )
            values_by_feature.update(scalar_results)
        chunk_results[collection_label] = values_by_feature
    return start, stop, chunk_results


def metric_chunk_worker(task):
    (
        original_name,
        compressed_name,
        variable,
        metrics,
        start,
        stop,
        ldcpy_path,
    ) = task
    ldcpy = _import_ldcpy(ldcpy_path)
    with xr.open_dataset(original_name) as original_dataset, xr.open_dataset(
        compressed_name
    ) as compressed_dataset:
        if variable not in original_dataset or variable not in compressed_dataset:
            raise KeyError(f"{variable!r} is missing from a metric source file")
        original, compressed = xr.align(
            original_dataset[variable], compressed_dataset[variable], join="inner"
        )
        available = min(
            original.sizes.get("time", 0), compressed.sizes.get("time", 0)
        )
        if available < stop:
            raise ValueError(
                f"Only {available} aligned timesteps are available; {stop} needed"
            )
        original = original.isel(time=slice(start, stop)).load()
        compressed = compressed.isel(time=slice(start, stop)).load()

    expected = stop - start
    results = {
        metric: np.empty(expected, dtype=np.float64) for metric in metrics
    }
    for local_index in range(expected):
        calculator = ldcpy.Diffcalcs(
            original.isel(time=local_index),
            compressed.isel(time=local_index),
            data_type="cam-fv",
            aggregate_dims=["lat", "lon"],
        )
        for metric in metrics:
            results[metric][local_index] = scalar_value(
                calculator.get_diff_calc(METRIC_CALCULATIONS[metric])
            )
    return start, stop, results


def run_feature_workers(
    source_specs, variable, timesteps, workers, ldcpy_path, output_dir, overwrite
):
    pending_specs = []
    destinations = {}
    for collection_label, source_path, features in source_specs:
        pending = []
        for feature in features:
            destination = feature_path(
                output_dir, variable, collection_label, feature, timesteps
            )
            if (
                destination.exists()
                and not overwrite
                and valid_existing_feature(destination, timesteps)
            ):
                logging.info("Skipping existing %s", destination)
            else:
                if destination.exists() and not overwrite:
                    logging.warning("Recomputing invalid existing %s", destination)
                pending.append(feature)
                destinations[(collection_label, feature)] = (
                    destination,
                    source_path,
                )
        if pending:
            pending_specs.append((collection_label, str(source_path), tuple(pending)))
    if not pending_specs:
        return

    chunks = chunk_ranges(timesteps, workers)
    assembled = {
        (label, feature): np.empty(timesteps, dtype=np.float64)
        for label, _, features in pending_specs
        for feature in features
    }
    tasks = [
        (pending_specs, variable, start, stop, ldcpy_path)
        for start, stop in chunks
    ]
    with ProcessPoolExecutor(
        max_workers=len(chunks), mp_context=get_context("spawn")
    ) as executor:
        futures = [executor.submit(feature_chunk_worker, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            start, stop, chunk_results = future.result()
            for label, values_by_feature in chunk_results.items():
                for feature, values in values_by_feature.items():
                    assembled[(label, feature)][start:stop] = values
            logging.info("Finished feature chunk %d/%d", completed, len(chunks))

    for key, values in assembled.items():
        destination, source_path = destinations[key]
        write_feature(
            values, destination, variable, key[1], source_path, timesteps
        )
        logging.info("Wrote %s", destination)


def run_metric_workers(
    original_path,
    compressed_path,
    variable,
    original_label,
    compressed_label,
    metrics,
    timesteps,
    workers,
    ldcpy_path,
    output_dir,
    overwrite,
):
    pending = []
    destinations = {}
    for metric in metrics:
        destination = metric_path(
            output_dir,
            variable,
            original_label,
            compressed_label,
            metric,
            timesteps,
        )
        if (
            destination.exists()
            and not overwrite
            and valid_existing_metric(destination, timesteps)
        ):
            logging.info("Skipping existing %s", destination)
        else:
            if destination.exists() and not overwrite:
                logging.warning("Recomputing invalid existing %s", destination)
            pending.append(metric)
            destinations[metric] = destination
    if not pending:
        return

    chunks = chunk_ranges(timesteps, workers)
    assembled = {
        metric: np.empty(timesteps, dtype=np.float64) for metric in pending
    }
    tasks = [
        (
            str(original_path),
            str(compressed_path),
            variable,
            tuple(pending),
            start,
            stop,
            ldcpy_path,
        )
        for start, stop in chunks
    ]
    with ProcessPoolExecutor(
        max_workers=len(chunks), mp_context=get_context("spawn")
    ) as executor:
        futures = [executor.submit(metric_chunk_worker, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            start, stop, chunk_results = future.result()
            for metric, values in chunk_results.items():
                assembled[metric][start:stop] = values
            logging.info("Finished metric chunk %d/%d", completed, len(chunks))

    for metric, values in assembled.items():
        write_metric(values, destinations[metric])
        logging.info("Wrote %s", destinations[metric])


def source_layout(config, compressions, variable):
    subdirectories = config["SubDirs"]
    prefixes = config["FilenamePre"]
    suffixes = config["FilenamePost"]
    if not (len(subdirectories) == len(prefixes) == len(suffixes)):
        raise ValueError("SubDirs, FilenamePre, and FilenamePost must have equal lengths")
    original_root = Path(config["OrigPath"])
    compressed_root = Path(config["CompPath"])
    compressed_sources = []
    for compression in compressions:
        for subdirectory, prefix, suffix in zip(
            subdirectories, prefixes, suffixes
        ):
            compressed_sources.append(
                (
                    f"{subdirectory}_{compression}",
                    compressed_root
                    / subdirectory
                    / compression
                    / f"{prefix}{variable}{suffix}",
                )
            )
    original_sources = []
    for subdirectory, prefix, suffix in zip(subdirectories, prefixes, suffixes):
        original_sources.append(
            (
                f"{subdirectory}_orig",
                original_root
                / subdirectory
                / "orig"
                / f"{prefix}{variable}{suffix}",
            )
        )
    return compressed_sources, original_sources


def combine_outputs(
    config,
    variables,
    features,
    requested_quality,
    feature_dir,
    metrics_dir,
    combined_output_dir,
    timesteps,
    overwrite,
):
    destination_dir = (
        Path(combined_output_dir).resolve()
        if combined_output_dir
        else output_dir / f"rf_inputs_time{timesteps}"
    )
    destination_dir.mkdir(parents=True, exist_ok=True)
    compressions = list(requested_quality)
    subdirectories = config["SubDirs"]
    feature_collections = [
        f"{subdirectory}_{compression}"
        for compression in compressions
        for subdirectory in subdirectories
    ] + [f"{subdirectory}_orig" for subdirectory in subdirectories]
    metric_names = []
    for metrics in requested_quality.values():
        for metric in metrics:
            if metric not in metric_names:
                metric_names.append(metric)

    for variable in variables:
        feature_destination = destination_dir / f"{variable}_features_time{timesteps}.nc"
        metric_destination = (
            destination_dir / f"{variable}_quality_metrics_time{timesteps}.nc"
        )
        if (
            feature_destination.exists()
            and metric_destination.exists()
            and not overwrite
        ):
            logging.info("Skipping existing outputs for %s", variable)
            continue

        feature_values = np.empty(
            (len(feature_collections), timesteps, len(features)), dtype=np.float64
        )
        for collection_index, collection_label in enumerate(feature_collections):
            for feature_index, feature in enumerate(features):
                path = feature_path(
                    feature_dir, variable, collection_label, feature, timesteps
                )
                if not path.is_file():
                    raise FileNotFoundError(path)
                with xr.open_dataarray(path) as feature_data:
                    values = np.asarray(feature_data.values).reshape(-1)
                if values.size != timesteps:
                    raise ValueError(
                        f"{path} contains {values.size} values; expected {timesteps}"
                    )
                feature_values[collection_index, :, feature_index] = values

        quality_values = np.full(
            (
                len(subdirectories),
                len(compressions),
                timesteps,
                len(metric_names),
            ),
            np.nan,
            dtype=np.float64,
        )
        for collection_index, subdirectory in enumerate(subdirectories):
            original_label = f"{subdirectory}_orig"
            for compression_index, compression in enumerate(compressions):
                compressed_label = f"{subdirectory}_{compression}"
                for metric in requested_quality[compression]:
                    metric_index = metric_names.index(metric)
                    path = metric_path(
                        metrics_dir,
                        variable,
                        original_label,
                        compressed_label,
                        metric,
                        timesteps,
                    )
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    values = np.asarray(np.load(path)).reshape(-1)
                    if values.size != timesteps:
                        raise ValueError(
                            f"{path} contains {values.size} values; expected {timesteps}"
                        )
                    quality_values[
                        collection_index, compression_index, :, metric_index
                    ] = values

        timestep = np.arange(1, timesteps + 1)
        feature_dataset = xr.Dataset(
            data_vars={
                "feature_values": (
                    ("collection", "timestep", "feature"),
                    feature_values,
                )
            },
            coords={
                "collection": feature_collections,
                "timestep": timestep,
                "feature": features,
            },
            attrs={
                "variable": variable,
                "timesteps": timesteps,
                "collection_order": "numeric ZFP level ascending, then original",
            },
        )
        metric_dataset = xr.Dataset(
            data_vars={
                "quality_metric_values": (
                    ("collection", "compression", "timestep", "metric"),
                    quality_values,
                )
            },
            coords={
                "collection": [f"{item}_orig" for item in subdirectories],
                "compression": compressions,
                "timestep": timestep,
                "metric": metric_names,
            },
            attrs={
                "variable": variable,
                "timesteps": timesteps,
                "compression_order": "numeric ZFP level ascending",
                "quality_requests": json.dumps(requested_quality),
            },
        )
        write_dataset(feature_dataset, feature_destination)
        write_dataset(metric_dataset, metric_destination)
        verify_combined_outputs(
            feature_destination,
            metric_destination,
            feature_collections,
            compressions,
            timesteps,
        )
        logging.info("Wrote %s and %s", feature_destination, metric_destination)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    config_path = Path(args.config).resolve()
    with config_path.open() as stream:
        config = json.load(stream)
    variables = select_variables(config, args.variable, args.variable_index)
    features = config["RFFeatureList"]
    all_quality = quality_requests(args.quality_config or [args.config])
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = (
        Path(args.metrics_dir).resolve() if args.metrics_dir else output_dir
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "combine":
        combine_outputs(
            config,
            variables,
            features,
            all_quality,
            output_dir,
            metrics_dir,
            args.combined_output_dir,
            args.timesteps,
            args.overwrite,
        )
        return
    if args.mode == "features" and args.compression_index is not None:
        raise ValueError("--compression-index cannot be used with --mode features")

    selected_quality = select_compression(all_quality, args.compression_index)
    ldcpy_path = config.get("OptLdcpyDevPath")
    started = time.perf_counter()
    for variable in variables:
        compressed_sources, original_sources = source_layout(
            config, list(all_quality), variable
        )
        if args.mode in ("both", "features"):
            sources = compressed_sources + original_sources
            for _, source_path in sources:
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
            logging.info(
                "Computing %d features for %d ordered collections of %s with %d workers",
                len(features),
                len(sources),
                variable,
                args.workers,
            )
            for collection_number, (label, path) in enumerate(sources, start=1):
                logging.info(
                    "Feature collection %d/%d: %s",
                    collection_number,
                    len(sources),
                    label,
                )
                run_feature_workers(
                    [(label, path, tuple(features))],
                    variable,
                    args.timesteps,
                    args.workers,
                    ldcpy_path,
                    output_dir,
                    args.overwrite,
                )

        if args.mode not in ("both", "metrics"):
            continue
        source_lookup = dict(compressed_sources)
        for subdirectory, prefix, suffix in zip(
            config["SubDirs"], config["FilenamePre"], config["FilenamePost"]
        ):
            original_label = f"{subdirectory}_orig"
            original_path = (
                Path(config["OrigPath"])
                / subdirectory
                / "orig"
                / f"{prefix}{variable}{suffix}"
            )
            if not original_path.is_file():
                raise FileNotFoundError(original_path)
            for compression, metrics in selected_quality.items():
                compressed_label = f"{subdirectory}_{compression}"
                compressed_path = source_lookup[compressed_label]
                if not compressed_path.is_file():
                    raise FileNotFoundError(compressed_path)
                logging.info(
                    "Computing %s for %s at %s with %d workers",
                    ",".join(metrics),
                    variable,
                    compression,
                    args.workers,
                )
                for metric in metrics:
                    logging.info("Metric checkpoint: %s", metric)
                    run_metric_workers(
                        original_path,
                        compressed_path,
                        variable,
                        original_label,
                        compressed_label,
                        [metric],
                        args.timesteps,
                        args.workers,
                        ldcpy_path,
                        metrics_dir,
                        args.overwrite,
                    )
    logging.info("Completed in %.1f seconds", time.perf_counter() - started)


if __name__ == "__main__":
    main()
