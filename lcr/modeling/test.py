import xarray as xr
import glob


def test():
    features = [
        "w_e_first_differences_max",
        "w_e_first_differences",
        "n_s_first_differences_max",
        "n_s_first_differences",
        "mean",
        "ew_con_var",
        "ns_con_var"
    ]

    # Loop over each feature and combine files
    for feature in features:
        # Define the filename pattern for the current feature, where only VARNAME changes
        file_pattern = f"*_combined_lens1_ens25_1920_orig_FEATURE_{feature}_all_time100_second.nc"

        # Find all files in the current directory matching this pattern
        file_list = glob.glob(file_pattern)

        # Skip if no files match the pattern for the current feature
        if not file_list:
            print(f"No files found for feature '{feature}', skipping.")
            continue

        # Load each file as an xarray DataArray
        data_arrays = [xr.open_dataarray(file) for file in file_list]

        # Combine all DataArrays along the 'sample' dimension
        combined_data = xr.concat(data_arrays, dim="sample")

        # Save the combined dataset with 'varname' set to "all"
        output_filename = f"all_combined_lens1_ens25_1920_orig_FEATURE_{feature}_all_time100_second.nc"
        combined_data.to_netcdf(output_filename)

        print(f"Files for feature '{feature}' have been combined and saved as '{output_filename}'")


# Now you can proceed with further operations on data_1 and data_2
if __name__ == "__main__":
    test()