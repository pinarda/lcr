import xarray as xr
import numpy as np

def labeling():
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
        "ns_con_var",
        "real_information_cutoff",
        "entropy",
        "magnitude_range"
    ]

    # Define the variable names in the desired order
    variables_order = ['FLNS', 'LHFLX', 'PRECSL', 'PRECT', 'PSL', 'Q200', 'Q500', 'Q850',
                       'SHFLX', 'T200', 'T500', 'T850', 'TAUX', 'TAUY', 'TREFHTMX', 'TS', 'U010']

    # Assume there are 100 timesteps for each variable
    timesteps_per_variable = 100

    # Repeat each variable name for each timestep
    new_sample_coord = np.repeat(variables_order, timesteps_per_variable)

    # Loop over each feature and process its corresponding file
    for feature in features:
        # Construct the filename based on the feature
        file_path = f"{data_dir}FLNS_LHFLX_PRECSL_PRECT_PSL_Q200_Q500_Q850_SHFLX_T200_T500_T850_TAUX_TAUY_TREFHTMX_TS_U010_combined_lens1_ens25_1920_orig_FEATURE_{feature}_all_time100_second.nc"

        try:
            # Load the combined DataArray for this feature
            combined_data = xr.open_dataarray(file_path)

            # Assign the new coordinate to the sample dimension
            combined_data = combined_data.assign_coords(sample=("sample", new_sample_coord))

            # Save the modified data with labeled samples for this feature
            output_filename = f"{data_dir}FLNS_LHFLX_PRECSL_PRECT_PSL_Q200_Q500_Q850_SHFLX_T200_T500_T850_TAUX_TAUY_TREFHTMX_TS_U010_labeled_combined_lens1_ens25_1920_orig_FEATURE_{feature}_all_time100_second.nc"
            combined_data.to_netcdf(output_filename)

            print(f"Data for feature '{feature}' has been labeled and saved as '{output_filename}'")

        except FileNotFoundError:
            print(f"File for feature '{feature}' not found, skipping.")


# Now you can proceed with further operations on data_1 and data_2
if __name__ == "__main__":
    labeling()