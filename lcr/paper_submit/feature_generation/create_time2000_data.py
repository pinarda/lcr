#!/usr/bin/env python3
"""Create a lossless archive of the source records used by the RF workflow."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


DEFAULT_PLUGIN_PATH = "/glade/work/haiyingx/H5Z-ZFP-PLUGIN-unbiased/plugin"
os.environ.setdefault("HDF5_PLUGIN_PATH", DEFAULT_PLUGIN_PATH)

import numpy as np
import xarray as xr


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Copy the exact first N positional timesteps used by feature generation "
            "into a restartable, losslessly stored source-data archive."
        )
    )
    parser.add_argument("--config", required=True, help="Primary model JSON config")
    parser.add_argument(
        "--quality-config",
        action="append",
        default=[],
        help=(
            "Config contributing compression directories; may be supplied more "
            "than once. Defaults to --config."
        ),
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="New root that will contain SubDirs/orig and SubDirs/zfp_p_X",
    )
    parser.add_argument("--timesteps", type=int, default=2000)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Maximum concurrent ncks processes (keep small because this is I/O-bound)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing subset files instead of validating and skipping them",
    )
    parser.add_argument(
        "--skip-variable",
        action="append",
        default=[],
        help="Variable to omit explicitly; may be repeated",
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


def compression_sort_key(name):
    match = re.search(r"(\d+)(?!.*\d)", name)
    if match:
        return (0, int(match.group(1)), name)
    return (1, 0, name)


def load_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def requested_compressions(paths):
    compressions = set()
    for path in paths:
        compressions.update(load_json(path)["CompDirs"])
    return sorted(compressions, key=compression_sort_key)


def source_records(config, compressions, output_root, skip_variables=()):
    variables = flatten_variables(config["VarList"])
    if len(variables) != len(set(variables)):
        raise ValueError("VarList contains duplicate variable names")
    unknown_skips = set(skip_variables) - set(variables)
    if unknown_skips:
        raise ValueError(
            f"Skipped variables are not present in VarList: {sorted(unknown_skips)}"
        )
    variables = [
        variable for variable in variables if variable not in skip_variables
    ]

    subdirectories = config["SubDirs"]
    prefixes = config["FilenamePre"]
    suffixes = config["FilenamePost"]
    if not (len(subdirectories) == len(prefixes) == len(suffixes)):
        raise ValueError("SubDirs, FilenamePre, and FilenamePost must have equal lengths")

    records = []
    for subdirectory, prefix, suffix in zip(subdirectories, prefixes, suffixes):
        for variable in variables:
            filename = f"{prefix}{variable}{suffix}"
            original = Path(config["OrigPath"]) / subdirectory / "orig" / filename
            records.append(
                {
                    "subdirectory": subdirectory,
                    "collection": "orig",
                    "variable": variable,
                    "source": original,
                    "destination": output_root / subdirectory / "orig" / filename,
                }
            )
            for compression in compressions:
                records.append(
                    {
                        "subdirectory": subdirectory,
                        "collection": compression,
                        "variable": variable,
                        "source": (
                            Path(config["CompPath"])
                            / subdirectory
                            / compression
                            / filename
                        ),
                        "destination": (
                            output_root / subdirectory / compression / filename
                        ),
                    }
                )
    return records, variables


def array_equal(left, right):
    return left.shape == right.shape and np.array_equal(left, right)


def dataset_signature(path, variable, timesteps):
    with xr.open_dataset(path, decode_cf=False, mask_and_scale=False) as dataset:
        if variable not in dataset:
            raise ValueError(f"{variable!r} is not present in {path}")
        data = dataset[variable]
        if "time" not in data.dims:
            raise ValueError(f"{variable!r} in {path} has no time dimension")
        available = data.sizes["time"]
        if available < timesteps:
            raise ValueError(
                f"{path} has only {available} time records; {timesteps} are required"
            )

        signature = {
            "dimensions": tuple(data.dims),
            "sizes": {
                dimension: (timesteps if dimension == "time" else data.sizes[dimension])
                for dimension in data.dims
            },
            "coordinates": {},
        }
        for dimension in data.dims:
            if dimension not in dataset.variables:
                continue
            coordinate = dataset[dimension]
            if coordinate.dims != (dimension,):
                continue
            if dimension == "time":
                values = coordinate.isel(time=slice(0, timesteps)).values
            else:
                values = coordinate.values
            signature["coordinates"][dimension] = np.asarray(values)
        return signature


def compare_signatures(reference, candidate, description):
    if reference["dimensions"] != candidate["dimensions"]:
        raise ValueError(f"Dimension order differs for {description}")
    if reference["sizes"] != candidate["sizes"]:
        raise ValueError(f"Dimension sizes differ for {description}")
    if reference["coordinates"].keys() != candidate["coordinates"].keys():
        raise ValueError(f"Coordinate variables differ for {description}")
    for name in reference["coordinates"]:
        if not array_equal(
            reference["coordinates"][name], candidate["coordinates"][name]
        ):
            raise ValueError(f"Coordinate {name!r} differs for {description}")


def preflight(records, timesteps):
    missing = [
        str(record["source"])
        for record in records
        if not record["source"].is_file()
    ]
    if missing:
        preview = "\n".join(missing[:20])
        remainder = len(missing) - min(len(missing), 20)
        suffix = f"\n... and {remainder} more" if remainder else ""
        raise FileNotFoundError(f"Missing {len(missing)} source files:\n{preview}{suffix}")

    original_signatures = {}
    source_signatures = {}
    for number, record in enumerate(records, start=1):
        key = (record["subdirectory"], record["variable"])
        signature = dataset_signature(
            record["source"], record["variable"], timesteps
        )
        source_signatures[str(record["source"])] = signature
        if record["collection"] == "orig":
            original_signatures[key] = signature
        else:
            compare_signatures(
                original_signatures[key],
                signature,
                f"{record['collection']} versus orig for {record['variable']}",
            )
        if number % 25 == 0 or number == len(records):
            print(f"Preflight checked {number}/{len(records)} source files", flush=True)
    return source_signatures


def valid_existing(record, source_signature, timesteps):
    destination = record["destination"]
    if not destination.is_file():
        return False
    try:
        output_signature = dataset_signature(
            destination, record["variable"], timesteps
        )
        with xr.open_dataset(
            destination, decode_cf=False, mask_and_scale=False
        ) as dataset:
            if dataset[record["variable"]].sizes["time"] != timesteps:
                return False
        compare_signatures(
            source_signature,
            output_signature,
            f"existing output {destination}",
        )
        return True
    except (OSError, ValueError):
        return False


def run_ncks(ncks, record, timesteps):
    destination = record["destination"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A stable name lets a later restart remove a file left by a walltime kill.
    temporary = destination.with_name(f".{destination.name}.partial")
    if temporary.exists():
        temporary.unlink()
    command = [
        ncks,
        "-O",
        "-4",
        "-L",
        "1",
        "-d",
        f"time,0,{timesteps - 1}",
        str(record["source"]),
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return record, temporary


def write_manifest(output_root, records, timesteps):
    destination = output_root / f"manifest_time{timesteps}.tsv"
    temporary = destination.with_name(f".{destination.name}.partial.{os.getpid()}")
    with temporary.open("w") as stream:
        stream.write("subdirectory\tcollection\tvariable\ttimesteps\tbytes\tsource\tdestination\n")
        for record in records:
            output = record["destination"]
            stream.write(
                "\t".join(
                    [
                        record["subdirectory"],
                        record["collection"],
                        record["variable"],
                        str(timesteps),
                        str(output.stat().st_size),
                        str(record["source"]),
                        str(output),
                    ]
                )
                + "\n"
            )
    os.replace(temporary, destination)
    return destination


def main():
    args = parse_args()
    if args.timesteps < 1:
        raise ValueError("--timesteps must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    ncks = shutil.which("ncks")
    if ncks is None:
        raise RuntimeError("ncks was not found; load the NCO module before running")

    config = load_json(args.config)
    quality_paths = args.quality_config or [args.config]
    compressions = requested_compressions(quality_paths)
    output_root = Path(args.output_root).resolve()
    records, variables = source_records(
        config,
        compressions,
        output_root,
        args.skip_variable,
    )

    print(f"Variables: {len(variables)}", flush=True)
    if args.skip_variable:
        print(f"Skipped variables: {', '.join(args.skip_variable)}", flush=True)
    print(f"Collections: orig, {', '.join(compressions)}", flush=True)
    print(f"Files: {len(records)}", flush=True)
    print(f"Destination: {output_root}", flush=True)
    print(f"Timesteps: positional indices 0-{args.timesteps - 1}", flush=True)

    source_signatures = preflight(records, args.timesteps)

    pending = []
    skipped = 0
    for record in records:
        source_signature = source_signatures[str(record["source"])]
        if record["destination"].exists() and not args.overwrite:
            if valid_existing(record, source_signature, args.timesteps):
                skipped += 1
                continue
            raise ValueError(
                f"Existing output is invalid: {record['destination']}. "
                "Remove it or rerun with --overwrite."
            )
        if record["source"].resolve() == record["destination"].resolve():
            raise ValueError(f"Source and destination are identical: {record['source']}")
        pending.append(record)

    print(
        f"Ready to create {len(pending)} files; {skipped} valid files will be skipped",
        flush=True,
    )
    completed = 0
    with ThreadPoolExecutor(max_workers=min(args.workers, len(pending) or 1)) as executor:
        futures = [
            executor.submit(run_ncks, ncks, record, args.timesteps)
            for record in pending
        ]
        for future in as_completed(futures):
            record, temporary = future.result()
            try:
                output_signature = dataset_signature(
                    temporary, record["variable"], args.timesteps
                )
                source_signature = source_signatures[str(record["source"])]
                compare_signatures(
                    source_signature,
                    output_signature,
                    f"new output {record['destination']}",
                )
                with xr.open_dataset(
                    temporary, decode_cf=False, mask_and_scale=False
                ) as dataset:
                    actual = dataset[record["variable"]].sizes["time"]
                if actual != args.timesteps:
                    raise ValueError(
                        f"{temporary} has {actual} timesteps; expected {args.timesteps}"
                    )
                os.replace(temporary, record["destination"])
            except Exception:
                if temporary.exists():
                    temporary.unlink()
                raise
            completed += 1
            print(
                f"Created {completed}/{len(pending)}: "
                f"{record['collection']}/{record['destination'].name}",
                flush=True,
            )

    manifest = write_manifest(output_root, records, args.timesteps)
    print(
        f"Complete: {len(records)} validated files ({len(pending)} created, "
        f"{skipped} reused). Manifest: {manifest}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
