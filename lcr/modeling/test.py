import xarray as xr
import glob
import re

def test():
    # Directory containing the data files
    data_dir = "data/"

    # List of features to process
    features = [
        "w_e_first_differences_max",
        "w_e_first_differences",
        "n_s_first_differences_max",
        "n_s_first_differences",
        "mean",
        "ew_con_var",
        "ns_con_var"
    ]

    # Regex pattern to extract varname from the filename
    varname_pattern = re.compile(r"([^/]+)_combined_lens1_ens25_1920_orig_FEATURE_")

    # Loop over each feature and combine files
    for feature in features:
        # Define the filename pattern for the current feature, with directory prefix
        file_pattern = f"{data_dir}*_combined_lens1_ens25_1920_orig_FEATURE_{feature}_all_time100_second.nc"

        # Find all files in the specified directory matching this pattern
        file_list = glob.glob(file_pattern)

        # Skip if no files match the pattern for the current feature
        if not file_list:
            print(f"No files found for feature '{feature}', skipping.")
            continue

        # Load each file as an xarray DataArray and store varnames in order
        data_arrays = []
        varnames = []
        for file in file_list:
            # Extract varname from the filename using the regex pattern
            match = varname_pattern.search(file)
            if match:
                varname = match.group(1)
                varnames.append(varname)
                data_arrays.append(xr.open_dataarray(file))

        # Combine all DataArrays along the 'sample' dimension
        combined_data = xr.concat(data_arrays, dim="sample")

        # Create a filename based on the concatenated varnames in order
        varname_str = "_".join(varnames)
        output_filename = f"{data_dir}{varname_str}_combined_lens1_ens25_1920_orig_FEATURE_{feature}_all_time100_second.nc"

        # Save the combined dataset
        combined_data.to_netcdf(output_filename)

        print(
            f"Files for feature '{feature}' have been combined in order of '{varname_str}' and saved as '{output_filename}'")


# Now you can proceed with further operations on data_1 and data_2
if __name__ == "__main__":
    test()