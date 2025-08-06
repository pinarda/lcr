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

    variables_order = [
        "FLNS", "FLNSC", "FSNS", "FSNSC", "LHFLX", "PRECL", "PRECSC", "PRECSL", "PRECT",
        "PSL", "Q200", "Q500", "Q850", "QBOT", "SHFLX", "T010", "T200", "T500", "T850",
        "TAUX", "TAUY", "TMQ", "TREFHT", "TREFHTMN", "TREFHTMX", "TS", "U010", "U200",
        "U500", "U850", "UBOT", "V200", "V500", "V850", "VBOT", "WSPDSRFAV", "Z050",
        "Z500", "bc_a1_SRF", "dst_a1_SRF", "dst_a3_SRF", "pom_a1_SRF", "so4_a1_SRF",
        "so4_a2_SRF", "so4_a3_SRF", "soa_a1_SRF", "soa_a2_SRF"
    ]

    # Define the variable names in the desired order
    # variables_order = ['FLNS', 'LHFLX', 'PRECSL', 'PRECT', 'PSL', 'Q200', 'Q500', 'Q850',
    #                    'SHFLX', 'T200', 'T500', 'T850', 'TAUX', 'TAUY', 'TREFHTMX', 'TS', 'U010']

    # Assume there are 100 timesteps for each variable
    timesteps_per_variable = 1600

    # Repeat each variable name for each timestep
    new_sample_coord = np.repeat(variables_order, timesteps_per_variable)

    # Loop over each feature and process its corresponding file
    for feature in features:
        # Construct the filename based on the feature
        file_path = f"{data_dir}TREFHTMX_TS_LHFLX_PRECSL_PRECT_PSL_Q200_Q500_Q850_SHFLX_T200_T500_T850_TAUX_TAUY_U010_FLNS_FLNSC_FSNS_FSNSC_PRECL_PRECSC_QBOT_T010_TMQ_TREFHT_TREFHTMN_U200_U500_U850_UBOT_V200_V500_V850_VBOT_WSPDSRFAV_Z050_Z500_bc_a1_SRF_dst_a1_SRF_dst_a3_SRF_pom_a1_SRF_so4_a1_SRF_so4_a2_SRF_so4_a3_SRF_soa_a1_SRF_soa_a2_SRF_combined_lens1_ens25_1920_orig_FEATURE_{feature}_all_time1600_second.nc"


        try:
            # Load the combined DataArray for this feature
            combined_data = xr.open_dataarray(file_path)

            # Assign the new coordinate to the sample dimension
            combined_data = combined_data.assign_coords(sample=("sample", new_sample_coord))

            # Save the modified data with labeled samples for this feature
            output_filename = f"{data_dir}TREFHTMX_TS_LHFLX_PRECSL_PRECT_PSL_Q200_Q500_Q850_SHFLX_T200_T500_T850_TAUX_TAUY_U010_FLNS_FLNSC_FSNS_FSNSC_PRECL_PRECSC_QBOT_T010_TMQ_TREFHT_TREFHTMN_U200_U500_U850_UBOT_V200_V500_V850_VBOT_WSPDSRFAV_Z050_Z500_bc_a1_SRF_dst_a1_SRF_dst_a3_SRF_pom_a1_SRF_so4_a1_SRF_so4_a2_SRF_so4_a3_SRF_soa_a1_SRF_soa_a2_SRF_combined_lens1_ens25_1920_orig_FEATURE_{feature}_all_time1600_second.nc"

            combined_data.to_netcdf(output_filename)

            print(f"Data for feature '{feature}' has been labeled and saved as '{output_filename}'")

        except FileNotFoundError:
            print(f"File for feature '{feature}' not found, skipping.")


# Now you can proceed with further operations on data_1 and data_2
if __name__ == "__main__":
    labeling()