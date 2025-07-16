#!/usr/bin/env python3
"""
print_netcdf_dim_len.py

Quick utility: print dimension lengths (and optionally a variable's shape)
from a NetCDF file using xarray.

Usage examples
--------------
# Print all dimension lengths
python print_netcdf_dim_len.py all_big_combined_lens1_ens25_1920_orig_FEATURE_entropy_all_time2000_second.nc

# Print just the 'sample' dim length
python print_netcdf_dim_len.py file.nc -d sample

# Print multiple dims
python print_netcdf_dim_len.py file.nc -d sample -d lat -d lon

# Print dims + show a variable's shape/dims
python print_netcdf_dim_len.py file.nc -v combined
"""
import argparse
import sys
import xarray as xr


def main() -> int:
    p = argparse.ArgumentParser(description="Print dimension lengths from a NetCDF file.")
    p.add_argument("path", help="Path to NetCDF file.")
    p.add_argument("-d", "--dim", action="append",
                   help="Dimension name to report. Repeatable. "
                        "If omitted, all dims are printed.")
    p.add_argument("-v", "--var", help="Also report shape/dims for this variable.")
    p.add_argument("--engine", choices=["netcdf4", "h5netcdf", "scipy", "zarr"],
                   help="Optional xarray engine override.")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="Print just the integer length when a single --dim is given.")
    args = p.parse_args()

    # Open dataset
    try:
        ds = xr.open_dataset(args.path, engine=args.engine) if args.engine else xr.open_dataset(args.path)
    except Exception as e:
        print(f"ERROR: cannot open {args.path}: {e}", file=sys.stderr)
        return 1

    # Report dims
    if args.dim:
        if len(args.dim) == 1 and args.quiet:
            d = args.dim[0]
            if d not in ds.dims:
                print(f"ERROR: dimension '{d}' not found.", file=sys.stderr)
                return 2
            print(ds.sizes[d])
        else:
            for d in args.dim:
                n = ds.sizes.get(d)
                if n is None:
                    print(f"{d}=<MISSING>", file=sys.stderr)
                else:
                    print(f"{d}={n}")
    else:
        # print all dims
        for d, n in ds.sizes.items():
            print(f"{d}={n}")

    # Optional variable report
    if args.var:
        if args.var in ds:
            v = ds[args.var]
            shape = "x".join(str(i) for i in v.shape)
            dims = ",".join(v.dims)
            print(f"var {args.var}: shape {shape}  dims ({dims})")
        else:
            print(f"WARNING: variable '{args.var}' not found.", file=sys.stderr)

    ds.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
