# main.py

import os
import sys
import json
import numpy as np
import xarray as xr
import logging
import gc
from dask.distributed import Client
import dask.array as da
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process a configuration JSON file.")
    parser.add_argument('-c', '--config', type=str, default='config.json',
                        help="Path to the configuration JSON file (default: config.json)")

    # Parse the arguments
    args = parser.parse_args()

    # Read the specified JSON configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Your existing code that processes 'config' goes here
    print("Loaded configuration:", config)

    # Extract parameters
    base_orig_path = config.get('OrigPath')          # e.g., "/Users/alex/git/ldcpy/data/cam-fv/orig/"
    base_comp_path = config.get('CompPath')          # e.g., "/Users/alex/git/ldcpy/data/cam-fv/"
    var_list = config.get('VarList')                 # ["TS", "PRECT"]
    filename_pre = config.get('FilenamePre')         # ["", ""]
    filename_post = config.get('FilenamePost')       # [".100days.nc", ".100days.nc"]
    sub_dirs = config.get('SubDirs')                 # ["ens1", "ens2"]
    comp_dirs = config.get('CompDirs')               # ["zfp_1.0", "zfp_1e-1", "zfp_1e-3"]
    opt_ldcpy_dev_path = config.get('OptLdcpyDevPath')  # "/Users/alex/git/ldcpy"
    times = config.get('Times')                      # [60]
    navg = config.get('Navg')                        # 1
    storage_loc = config.get('StorageLoc')           # "./data/"
    stride = config.get('Stride')                    # 1
    cut_dataset = config.get('CutDataset')            # 0
    metric = config.get('Metric')                    # ["dssim", "pcc", "spre"]
    save_dir = config.get('SaveDir')                  # "/Users/alex/git/lcr/lcr/data_analysis/RFnew/plots/"
    modeltype = config.get('ModelType')               # "cnn"
    featurelist = config.get('RFFeatureList')           # ["mean", "ns_con_var"]
    flat_var_list =[var for group in var_list for var in group]

    # let's grab start time here
    start_time = pd.Timestamp.now()

    # log the flat var list
    logging.info(f"Flat var list: {flat_var_list}")

    # Add ldcpy to sys.path
    if opt_ldcpy_dev_path:
        sys.path.insert(0, opt_ldcpy_dev_path)

    import ldcpy
    from training import get_data_labels, train_cnn, evaluate_model
    from classification_labels import compare_across_metrics


    # Define the custom labeling function
    def generate_classification_labels(metrics_info, metrics_data, compression_level_order):
        """
        Generates classification labels based on metrics_info and metrics_data.

        Parameters:
            metrics_info (dict): Dictionary containing 'comparison' and 'threshold' for each metric.
            metrics_data (dict): Nested dictionary containing metric values for each compression level.

        Returns:
            final_comparison_labels_combined (xarray.DataArray): Combined classification labels.
            final_labels_dict (dict): Dictionary of labels per metric.
        """
        final_labels_dict = {}

        for metric, info in metrics_info.items():
            logging.info(f"Processing metric: {metric}")

            labels = []
            compression_levels = []

            metric_values_dict = metrics_data.get(metric, {})
            if not metric_values_dict:
                logging.warning(f"No data for metric '{metric}'. Skipping.")
                continue

            for comp_label, metric_values in metric_values_dict.items():
                # Ensure comp_label is a string
                if not isinstance(comp_label, str):
                    logging.warning(f"Compression label '{comp_label}' is not a string. Converting to string.")
                    comp_label = str(comp_label)

                compression_levels.append(comp_label)

                # Convert metric_values to a DataArray
                data_da = xr.DataArray(metric_values)

                # Apply comparison and threshold
                if info['comparison'] == 'gt':
                    label_da = data_da >= info['threshold']
                elif info['comparison'] == 'lt':
                    label_da = data_da < info['threshold']
                else:
                    logging.error(f"Invalid comparison '{info['comparison']}' for metric '{metric}'. Skipping.")
                    continue

                labels.append(label_da)

            if not labels:
                logging.warning(f"No labels generated for metric '{metric}'.")
                final_labels_dict[metric] = None
                continue

            # Initialize final_labels with "None"
            final_labels = xr.full_like(labels[0], "None", dtype="object")

            for comp_label, label_da in zip(compression_levels, labels):
                final_labels = xr.where(
                    (final_labels == "None") & label_da,
                    comp_label,
                    final_labels
                )

            final_labels_dict[metric] = final_labels

        # Compare results across all metrics
        # Here, you can implement your own logic or use the existing compare_across_metrics function
        # For simplicity, we'll prioritize 'dssim' if available

        # Create an array to hold the final conservative label for each element
        combined_final_labels = xr.full_like(final_labels_dict[next(iter(final_labels_dict))], "None", dtype="object")

        for comp_level in compression_level_order[::-1]:
            for metric, label_da in final_labels_dict.items():
                if label_da is None:
                    continue
                # Update combined_final_labels with the most conservative available label
                combined_final_labels = xr.where(
                    (combined_final_labels == "None") & (label_da == comp_level),
                    comp_level,
                    combined_final_labels
                )

        # Ensure no elements remain labeled as "None", fill them with the most conservative option
        combined_final_labels = xr.where(
            combined_final_labels == "None",
            compression_level_order[-1],  # Fallback to the most conservative compression level
            combined_final_labels
        )

        final_comparison_labels_combined = combined_final_labels

        return final_comparison_labels_combined, final_labels_dict

    # Initialize dictionaries to hold files and labels for each variable
    # Mappings for prefixes and postfixes based on ensembles

    # Initialize dictionaries to store files and labels
    # files_dict = {varname: [] for varname in flat_var_list}

    # flat_var_list = [aitem for sublist in flat_var_list for item in sublist]
    files_dict = {varname: [] for varname in flat_var_list}
    labels_dict = {varname: [] for varname in flat_var_list}
    print(files_dict)

    i = 0
    for subdir in sub_dirs:

        pre = filename_pre[i]
        post = filename_post[i]
        i = i + 1

        for varname in flat_var_list:
            # Original data files and labels
            orig_file = os.path.join(base_orig_path, subdir, 'orig', pre + varname + post)
            files_dict[varname].append(orig_file)
            labels_dict[varname].append(f"{subdir}_orig")

            # Compressed data files and labels
            for comp_dir in comp_dirs:
                comp_file = os.path.join(base_comp_path, subdir, comp_dir, pre + varname + post)
                files_dict[varname].append(comp_file)
                labels_dict[varname].append(f"{subdir}_{comp_dir}")

    # Open datasets separately for each variable
    opened_datasets = {}

    client = Client(n_workers=4, threads_per_worker=1, memory_limit='20GB')  # Adjust as needed
    try:
        for varname in flat_var_list:
            logging.info(f"Opening datasets for variable: {varname}")

            # Retrieve files and labels for the current variable
            files = files_dict[varname]
            labels = labels_dict[varname]

            # Set varnames list to contain only the current variable
            varnames = [varname] * len(files)  # Each file has its varname

            # Open the datasets using ldcpy
            data_type = 'cam-fv'

            logging.info("Opening datasets with ldcpy.open_datasets...")
            dataset_col = ldcpy.open_datasets(
                list_of_files=files,
                labels=labels,
                data_type=data_type,
                varnames=varnames
            ).chunk(time=100)  # Adjust chunk size as needed\
            # Store the opened dataset
            opened_datasets[varname] = dataset_col
    except Exception as e:
        logging.error(f"An error occurred: {e}")

    # cut the data here to only use times[0] number of time steps
    opened_datasets = {varname: ds.isel(time=slice(times[0])) for varname, ds in opened_datasets.items()}


    # Combine the datasets for all variables
    logging.info("Combining datasets for all variables...")

    # Initialize an empty list to collect datasets
    datasets_to_merge = []

    for varname in flat_var_list:
        dataset = opened_datasets[varname]
        datasets_to_merge.append(dataset)

    # Merge the datasets along the variable dimension
    combined_dataset = xr.merge(datasets_to_merge)

    logging.info("Datasets combined successfully.")

    # Initialize a nested dictionary to store metrics per metric and compression level
    metrics_data = {m: {} for m in metric}

    # Define metrics_info based on your configuration
    # This should include 'comparison' and 'threshold' for each metric
    # Example:
    metrics_info = {
        'dssim': {'comparison': 'gt', 'threshold': 0.995},
        'pcc': {'comparison': 'gt', 'threshold': 0.9995},
        'spre': {'comparison': 'lt', 'threshold': 95},
        'ks': {'comparison': 'lt', 'threshold': 0.05}
    }

    metrics_data = {varname: {m: {} for m in metric} for varname in flat_var_list}  # Separate metrics_data for each variable

    for varname in flat_var_list:
        logging.info(f"Computing metrics for variable: {varname}")

        dataset_col = opened_datasets[varname]

        # Prepare labels
        orig_labels = [f"{subdir}_orig" for subdir in sub_dirs]
        comp_labels = [f"{subdir}_{comp_dir}" for comp_dir in comp_dirs for subdir in sub_dirs]

        label_mapping = {f"{subdir}_{comp_dir}": f"{subdir}_orig" for comp_dir in comp_dirs for subdir in sub_dirs}

        # log the original labels
        logging.info(f"Original labels: {orig_labels}")

        # Iterate over compression directories and compute metrics
        for comp_label in comp_labels:
            orig_label = label_mapping.get(comp_label)
            if orig_label is None:
                logging.error(f"No matching orig_label found for comp_label '{comp_label}'. Skipping.")
                continue

            # Proceed with the logic using the orig_label and comp_label
            logging.info(f"Processing: orig_label={orig_label}, comp_label={comp_label}")

            for m in metric:
                # Create a unique filename based on varname, orig_label, comp_label, and metric
                metric_filename = f"{storage_loc}/{varname}_{orig_label}_{comp_label}_{m}_time{times[0]}_second.npy"
                # metric_filename = f"{storage_loc}/{varname}_{orig_label}_{comp_label}_{m}.npy"

                # Check if the file already exists
                if os.path.exists(metric_filename):
                    logging.info(f"Loading cached metric: {metric_filename}")
                    metrics_data[varname][m][comp_label] = np.load(metric_filename)
                    continue

                logging.info(f"Computing metrics between {orig_label} and {comp_label} for {m}...")

                # log all the collections in the dataset
                logging.info(f"Dataset collections: {dataset_col[varname]['collection'].values}")


                try:
                    ds_orig = dataset_col[varname].sel(collection=orig_label)
                    ds_comp = dataset_col[varname].sel(collection=comp_label)
                except KeyError as e:
                    logging.warning(f"Warning: {e}. Skipping this pair.")
                    continue

                # Ensure time alignment
                ds_orig_aligned, ds_comp_aligned = xr.align(ds_orig, ds_comp, join='inner')

                # Get the name of the time dimension
                time_dim_name = 'time'  # Replace 'time' with the actual name if different

                # Determine the number of time steps
                if time_dim_name not in ds_orig_aligned.dims:
                    logging.error(f"Error: Time dimension '{time_dim_name}' not found in datasets. Skipping this pair.")
                    continue

                num_time_steps = times[0]

                metric_values = []

                # log the number of time steps
                logging.info(f"Number of time steps: {num_time_steps}")

                # Iterate over each time step
                for t in range(num_time_steps):
                    # log the time step
                    logging.info(f"Processing time step {t}...")
                    # Select the current time slice for original and compressed datasets
                    ds_orig_time = ds_orig_aligned.isel(**{time_dim_name: t})
                    ds_comp_time = ds_comp_aligned.isel(**{time_dim_name: t})

                    # Create a new Diffcalcs object for the current time slice
                    dc = ldcpy.Diffcalcs(ds_orig_time, ds_comp_time, data_type='cam-fv', aggregate_dims=['lat', 'lon'])

                    # Compute metrics for each type
                    if m == 'dssim':
                        dssim_value = dc.get_diff_calc('ssim_fp')
                        if np.isscalar(dssim_value):
                            dssim_values = [dssim_value]
                        else:
                            dssim_values = dssim_value.values.flatten()
                        metric_values.extend(dssim_values)

                    elif m == 'pcc':
                        pcc_values = dc.get_diff_calc('pearson_correlation_coefficient').values.flatten()
                        metric_values.extend(pcc_values)

                    elif m == 'spre':
                        spre_values = dc.get_diff_calc('spatial_rel_error').values.flatten()
                        metric_values.extend(spre_values)
                    elif m == 'ks':
                        ks_values = dc.get_diff_calc('ks_p_value').values.flatten()
                        metric_values.extend(ks_values)

                # Save the computed metric to a file for future use
                np.save(metric_filename, metric_values)
                metrics_data[varname][m][comp_label] = metric_values

                logging.info(f"Saved metric data to {metric_filename}")



    # Combine metrics data from 'ens1' and 'ens2' under each compression level
    metrics_data_combined = {}
    for varname in flat_var_list:
        metrics_data_var = metrics_data[varname]
        metrics_data_combined[varname] = {}
        for metric_name in metrics_data_var:
            metrics_data_combined[varname][metric_name] = {}
            comp_level_dict = {}
            for comp_label in metrics_data_var[metric_name]:
                # comp_label is like 'ens1_zfp_1.0', 'ens2_zfp_1.0', etc.
                parts = comp_label.split('_')

                # Find the index of "zfp" in parts
                try:
                    zfp_index = parts.index('zfp')
                except ValueError:
                    logging.error(f"'zfp' not found in comp_label '{comp_label}'. Skipping.")
                    continue

                # Extract 'ens' part (everything before "zfp")
                ens = '_'.join(parts[:zfp_index])

                # Combine all elements starting with "zfp" into comp_level
                comp_level = '_'.join(parts[zfp_index:])  # e.g., 'zfp_1.0'
                if comp_level not in comp_level_dict:
                    comp_level_dict[comp_level] = []
                data_array = metrics_data_var[metric_name][comp_label]
                comp_level_dict[comp_level].append(data_array)
            # Now, for each comp_level, combine the arrays
            for comp_level in comp_level_dict:
                combined_array = np.concatenate(comp_level_dict[comp_level])
                metrics_data_combined[varname][metric_name][comp_level] = combined_array

    # Now use metrics_data_combined[varname] in generate_classification_labels
    final_comparison_labels_dict = {}
    final_labels_dict = {}

    for varname in flat_var_list:
        metrics_data_var = metrics_data_combined[varname]
        # here

        comparison_list = comp_dirs

        # acually, log it
        logging.info(f"Metrics data var: {metrics_data_var}")
        logging.info(f"Comparison list: {comparison_list}")
        logging.info(f"Metrics info: {metrics_info}")

        final_comparison_labels, final_labels = generate_classification_labels(metrics_info, metrics_data_var, comparison_list)
        final_comparison_labels_dict[varname] = final_comparison_labels
        final_labels_dict[varname] = final_labels

    logging.info("Classification labels generated successfully for all variables.")

    # Initialize list to store the reshaped data for each variable
    all_data_reshaped = []

    # Initialize list to store combined labels for each variable
    combined_labels_list = []

    for varname in flat_var_list:
        logging.info(f"Processing variable: {varname}")

        # Extract data for this variable
        data_var = opened_datasets[varname][varname]  # Get the DataArray for the variable

        # Select both 'ens1_orig' and 'ens2_orig' datasets
        data_var = data_var.sel(collection=[label for label in data_var['collection'].values if 'orig' in label])

        # log the number of time steps
        logging.info(f" length of data_var time {len(data_var['time'])} and collection {len(data_var['collection'])}")
        # cut the dataset to only use times[0] number of time steps
        # data_var = data_var.isel(time=slice(times[0]))

        # let's log all the labels
        logging.info(f"Labels for {varname}: {data_var['collection'].values}")

        # Stack 'collection' and 'time' into 'sample'
        # data_var = data_var.stack(sample=('collection', 'time')).reset_index('sample', drop=True)
        data_var = data_var.stack(sample=('collection', 'time')).drop_vars('collection')
        data_var = data_var.assign_coords(sample=data_var['sample'])
        data_var = data_var.reset_index('sample', drop=True)

        # Expand 'variable' dimension to size 1 and assign variable name
        data_var = data_var.assign_coords(variable=varname)
        # Add data to list
        all_data_reshaped.append(data_var)

        # Get labels for this variable
        labels_var = final_comparison_labels_dict[varname]

        logging.info(f"length of labels_var {len(labels_var)}")
        # Ensure labels are a flat array
        # first, convert to numpy array
        labels_var = labels_var.values
        labels_var = labels_var.flatten()
        combined_labels_list.append(labels_var)

    # Concatenate data along 'sample' dimension
    combined_data_reshaped = xr.concat(all_data_reshaped, dim='sample')

    # Concatenate labels
    combined_labels = np.concatenate(combined_labels_list)

    # Convert to xarray Dataset
    dataset_xr = xr.Dataset({'combined': combined_data_reshaped})

    # Ensure that the number of samples matches the number of labels
    if dataset_xr.dims['sample'] != combined_labels.size:
        raise ValueError(
            f"Number of samples ({dataset_xr.dims['sample']}) does not match number of labels ({combined_labels.size})."
        )

    logging.info("MLData prepared successfully.")

    # Number of variables
    nvars = len(flat_var_list)  # Should be equal to len(var_list)

    # Call train_cnn
    logging.info("Calling train_cnn for training...")

    jobid = 0
    j=0
    storageloc = storage_loc
    time = times[0]

    # get time here and log it with a description
    end_time = pd.Timestamp.now()
    logging.info(f"Time taken for data preparation: {end_time - start_time}")

    if modeltype == 'rf':
        # Feature computation and data loading
        features_np = compute_features(dataset_xr, featurelist, storage_loc, ','.join(flat_var_list) + '_combined',
                                       orig_label, "all", times)
        labels_np = np.array(combined_labels)
        features_np = features_np.T  # Transpose features

        # also get time here and log it with a description
        end_time = pd.Timestamp.now()
        logging.info(f"Time taken for feature computation: {end_time - start_time}")


        # Prepare data
        get_data_labels(
            dataset=features_np,
            labels=labels_np,
            time=times[0],
            varname=','.join(flat_var_list) + '_combined',
            nvar=nvars,
            storageloc=storage_loc,
            testset='1var',
            j=0,
            plotdir=save_dir,
            window_size=11,
            only_data=False,
            modeltype=modeltype,
            feature=None,
            featurelist=None,
            transform='quantile',
            jobid=0,
            cut_windows=False,
            metric=metric,
            vars=var_list
        )

        train_data_np = np.load(f"{storageloc}/train_data_{j}{time}{modeltype}{jobid}.npy")
        val_data_np = np.load(f"{storageloc}/val_data_{j}{time}{modeltype}{jobid}.npy")
        test_data_np = np.load(f"{storageloc}/test_data_{j}{time}{modeltype}{jobid}.npy")
        train_labels_np = np.load(f"{storageloc}/train_labels_{j}{time}{modeltype}{jobid}.npy")
        val_labels_np = np.load(f"{storageloc}/val_labels_{j}{time}{modeltype}{jobid}.npy")
        test_labels_np = np.load(f"{storageloc}/test_labels_{j}{time}{modeltype}{jobid}_{var_list[0]}.npy")
        label_encoder = np.load(f"{storageloc}/label_encoder_{j}{time}{modeltype}{jobid}_{var_list[0]}.pkl",
                                allow_pickle=True)

        # --- Random Forest ---
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)
        rf_model.fit(train_data_np, train_labels_np)

        # Evaluate on validation data
        val_predictions_rf = rf_model.predict(val_data_np)
        val_accuracy_rf = accuracy_score(val_labels_np, val_predictions_rf)
        print(f"Random Forest Validation Accuracy: {val_accuracy_rf}")

        # Evaluate on test data
        test_predictions_rf = rf_model.predict(test_data_np)
        test_accuracy_rf = accuracy_score(test_labels_np, test_predictions_rf)
        print(f"Random Forest Test Accuracy: {test_accuracy_rf}")

        # Confusion matrix and classification report for Random Forest
        cm_rf = confusion_matrix(test_labels_np, test_predictions_rf)
        cr_rf = classification_report(test_labels_np, test_predictions_rf)
        print(cm_rf)
        print(cr_rf)

        # Ensure all classes are present in the confusion matrix
        num_classes = len(label_encoder.classes_)
        cm_full = np.zeros((num_classes, num_classes), dtype=int)

        # Populate the confusion matrix with the actual results
        for i in range(cm_rf.shape[0]):
            for j in range(cm_rf.shape[1]):
                cm_full[i, j] = cm_rf[i, j]

        # Feature Importances for Random Forest
        rf_feature_importances = rf_model.feature_importances_
        np.save(f"{storageloc}/feature_importances_rf_{j}{time}{modeltype}{jobid}_{var_list[0]}.npy",
                rf_feature_importances)

        # Save Random Forest predictions and metrics
        np.save(f"{storageloc}/test_predictions_rf_{j}{time}{modeltype}{jobid}_{var_list[0]}.npy", test_predictions_rf)
        np.save(f"{storageloc}/classification_report_rf_{j}{time}{modeltype}{jobid}_{var_list[0]}.npy", cr_rf)

        # Save Confusion Matrix for Random Forest
        rf_confusion_df = pd.DataFrame(cm_full, index=label_encoder.classes_, columns=label_encoder.classes_)
        rf_confusion_df.to_csv(f"{storageloc}/confusion_matrix_rf_{j}{time}{modeltype}{jobid}_{var_list[0]}.csv")

        # --- Decision Tree ---
        print("Training Decision Tree...")
        dt_model = DecisionTreeClassifier(max_depth=10, random_state=0)
        dt_model.fit(train_data_np, train_labels_np)

        # Evaluate on validation data
        val_predictions_dt = dt_model.predict(val_data_np)
        val_accuracy_dt = accuracy_score(val_labels_np, val_predictions_dt)
        print(f"Decision Tree Validation Accuracy: {val_accuracy_dt}")

        # Evaluate on test data
        test_predictions_dt = dt_model.predict(test_data_np)
        test_accuracy_dt = accuracy_score(test_labels_np, test_predictions_dt)
        print(f"Decision Tree Test Accuracy: {test_accuracy_dt}")

        # Confusion matrix and classification report for Decision Tree
        cm_dt = confusion_matrix(test_labels_np, test_predictions_dt)
        cr_dt = classification_report(test_labels_np, test_predictions_dt)
        print(cm_dt)
        print(cr_dt)

        # Ensure all classes are present in the confusion matrix
        num_classes = len(label_encoder.classes_)
        cm_dt_full = np.zeros((num_classes, num_classes), dtype=int)

        # Populate the confusion matrix with the actual results
        for i in range(cm_rf.shape[0]):
            for j in range(cm_rf.shape[1]):
                cm_full[i, j] = cm_rf[i, j]

        # Plot Decision Tree
        plt.figure(figsize=(20, 10))
        plot_tree(
            dt_model,
            feature_names=train_data_np.columns if hasattr(train_data_np, "columns") else None,
            class_names=[str(cls) for cls in dt_model.classes_],
            filled=True,
            rounded=True,
            fontsize=10
        )
        plt.title("Decision Tree Visualization")
        plt.savefig(f"{storageloc}/decision_tree_{j}{time}{modeltype}{jobid}_{var_list[0]}.png")
        plt.show()

        # Save Decision Tree predictions and metrics
        np.save(f"{storageloc}/test_predictions_dt_{j}{time}{modeltype}{jobid}_{var_list[0]}.npy", test_predictions_dt)
        np.save(f"{storageloc}/classification_report_dt_{j}{time}{modeltype}{jobid}_{var_list[0]}.npy", cr_dt)

        # Save Confusion Matrix for Decision Tree
        dt_confusion_df = pd.DataFrame(cm_dt_full, index=label_encoder.classes_, columns=label_encoder.classes_)
        dt_confusion_df.to_csv(f"{storageloc}/confusion_matrix_dt_{j}{time}{modeltype}{jobid}_{var_list[0]}.csv")

        # Feature Importances for Decision Tree
        dt_feature_importances = dt_model.feature_importances_
        np.save(f"{storageloc}/feature_importances_dt_{j}{time}{modeltype}{jobid}_{var_list[0]}.npy",
                dt_feature_importances)
        # log the time taken for the entire process
        end_time = pd.Timestamp.now()
        logging.info(f"Time taken for Random Forest and Decision Tree training: {end_time - start_time}")

    else:
        # log time and start the process
        end_time = pd.Timestamp.now()
        logging.info(f"Time taken for data preparation: {end_time - start_time}")

        get_data_labels(
            dataset=dataset_xr,
            labels=combined_labels,
            time=times[0],  # Adjust based on train_cnn requirements
            varname=','.join(flat_var_list) + '_combined',
            nvar=nvars,
            storageloc=storage_loc,
            testset='1var',
            j=0,
            plotdir=save_dir,
            window_size=11,  # As per WINDOWSIZE
            only_data=False,
            modeltype='cnn',
            feature=None,
            featurelist=None,
            transform='quantile',
            jobid=0,
            cut_windows=False,
            metric=metric,
            vars = var_list
        )

        # stop the client
        client.close()

    # load all this data:
    # np.save(f"{storageloc}train_data_{j}{time}{modeltype}{jobid}.npy", train_data_np)
    # np.save(f"{storageloc}val_data_{j}{time}{modeltype}{jobid}.npy", val_data_np)
    # np.save(f"{storageloc}test_data_{j}{time}{modeltype}{jobid}.npy", test_data_np)
    # np.save(f"{storageloc}train_labels_{j}{time}{modeltype}{jobid}.npy", train_labels_np)
    # np.save(f"{storageloc}val_labels_{j}{time}{modeltype}{jobid}.npy", val_labels_np)
    # np.save(f"{storageloc}test_labels_{j}{time}{modeltype}{jobid}.npy", test_labels_np)


    if modeltype == "cnn":
        #log time and start the process
        end_time = pd.Timestamp.now()
        logging.info(f"Time taken for data preparation: {end_time - start_time}")
        train_data_np = np.load(f"{storageloc}/train_data_{j}{time}{modeltype}{jobid}.npy")
        val_data_np = np.load(f"{storageloc}/val_data_{j}{time}{modeltype}{jobid}.npy")
        test_data_np = np.load(f"{storageloc}/test_data_{j}{time}{modeltype}{jobid}.npy")
        train_labels_np = np.load(f"{storageloc}/train_labels_{j}{time}{modeltype}{jobid}.npy")
        val_labels_np = np.load(f"{storageloc}/val_labels_{j}{time}{modeltype}{jobid}.npy")
        test_labels_np = np.load(f"{storageloc}/test_labels_{j}{time}{modeltype}{jobid}_{var_list[0]}.npy")
        label_encoder = np.load(f"{storageloc}/label_encoder_{j}{time}{modeltype}{jobid}_{var_list[0]}.pkl", allow_pickle=True)

        # print the label encodings
        logging.info(f"Label encodings: {label_encoder.classes_}")

        # Expand the dimensions of the data arrays to include a channels dimension
        train_data_np = np.expand_dims(train_data_np, axis=-1)
        val_data_np = np.expand_dims(val_data_np, axis=-1)
        test_data_np = np.expand_dims(test_data_np, axis=-1)

        # Adjust the labels if necessary (ensure they are integers starting from 0)
        train_labels_np = train_labels_np.astype(int)
        val_labels_np = val_labels_np.astype(int)
        test_labels_np = test_labels_np.astype(int)


        # Call the function to train the model
        model = train_cnn(
            train_data_np,
            train_labels_np,
            val_data_np,
            val_labels_np,
            test_data_np,
            test_labels_np,
            modeltype="cnn",
            transform=None,
        )

        # log time and start the process
        end_time = pd.Timestamp.now()

        accuracy, confusion, classreport, test_predictions = evaluate_model(model, test_data_np, test_labels_np)

        # save the test predictions
        np.save(f"{storageloc}/test_predictions_{j}{time}{modeltype}{jobid}_{var_list[0]}.npy", test_predictions)

        # Map numerical predictions back to original labels
        predicted_labels = label_encoder.inverse_transform(test_predictions)

        # save the predictions
        np.save(f"{storageloc}/predictions_{j}{time}{modeltype}{jobid}_{var_list[0]}.npy", predicted_labels)

        # Identify unique labels present in predictions
        unique_labels = np.unique(predicted_labels)

        # Define all possible labels
        all_labels = list(label_encoder.classes_)

        # Set up confusion matrix DataFrame with both labels, filling missing values with 0
        confusion_df = pd.DataFrame(
            confusion,
            index=unique_labels,
            columns=unique_labels
        ).reindex(index=all_labels, columns=all_labels, fill_value=0)

        # save the confusion matrix and classification report
        np.save(f"{storageloc}/classification_report_{j}{time}{modeltype}{jobid}_{var_list[0]}.npy", classreport)
        # save the confusion matrix df as csv
        confusion_df.to_csv(f"{storageloc}/confusion_matrix_{j}{time}{modeltype}{jobid}_{var_list[0]}.csv")
        # save the label encoder classes
        np.save(f"{storageloc}/label_encoder_{j}{time}{modeltype}{jobid}_{var_list[0]}.pkl", label_encoder.classes_)
        logging.info(f"Time taken for CNN training: {end_time - start_time}")


    # make a plot of the confusion matrix where the height is the sum of the row, and the diagonal element in the row is shaded in a different color

    if modeltype == "cnn":
        confusion_df = pd.read_csv(f"{storageloc}/confusion_matrix_{j}{time}{modeltype}{jobid}_{var_list[0]}.csv", index_col=0)
    elif modeltype == "rf":
        confusion_df = pd.read_csv(f"{storageloc}/confusion_matrix_rf_{j}{time}{modeltype}{jobid}_{var_list[0]}.csv", index_col=0)
    # Calculate the column sums
    col_sums = confusion_df.sum(axis=0)

    # Extract diagonal (correct predictions) and off-diagonal values for plotting
    diagonal = np.diag(confusion_df)
    off_diagonal = col_sums - diagonal

    # Plotting
    fig, ax = plt.subplots()

    # label_encoder = np.load(f"{storageloc}/label_encoder_{j}{time}{modeltype}{jobid}_{var_list[0]}.pkl", allow_pickle=True)
    # all_labels = list(label_encoder.classes_)

    all_labels = list(confusion_df.index)

    # Plot correct predictions in bold color
    ax.bar(all_labels, diagonal, label="Correct Predictions", color="blue", edgecolor="black")

    # Plot misclassifications in a lighter color, stacked on top of correct predictions
    ax.bar(all_labels, off_diagonal, bottom=diagonal, label="Misclassifications", color="lightblue", edgecolor="black")

    # Adding labels and legend
    ax.set_xlabel("Labels")
    ax.set_ylabel("Counts")
    ax.set_title("Confusion Matrix Summary")
    ax.legend()

    plt.show()
    # save the plot
    fig.savefig(f"{storageloc}/confusion_matrix_plot_{j}{time}{modeltype}{jobid}_{var_list[0]}.png")



def compute_features(data_xr, featurelist, storage_loc="./data", varname="combined", orig_label="orig", m="metric", times=None):
    """
    Compute features for each sample in the dataset using ldcpy.

    Parameters:
        data_xr (xr.Dataset): The dataset containing the data samples.
        featurelist (list of str): List of features to compute using ldcpy.

    Returns:
        np.ndarray: Array of computed features for each sample.
    """
    import xarray as xr
    import ldcpy
    import numpy as np

    features_list = []
    n_samples = data_xr.dims['sample']

    # for i in range(n_samples):
    # sample = data_xr.isel(sample=i)  # Get the i-th sample
    sample = data_xr  # Get the i-th sample

    sample_da = sample['combined']   # Assuming 'combined' is the variable name

    # Create ldcpy Datasetcalcs object
    dc = ldcpy.Datasetcalcs(
        sample_da, "cam-fv", ["lat", "lon"], weighted=False
    )

    sample_features = []
    for feature in featurelist:
        # check if the file already exists
        if os.path.exists(f"{storage_loc}/all_big_combined_{orig_label}_FEATURE_{feature}_all_time1600_second.nc"):
            logging.info(f"Loading cached feature: {feature}")
            feat_da = xr.open_dataarray(f"{storage_loc}/all_big_combined_{orig_label}_FEATURE_{feature}_all_time1600_second.nc")

            # varname = "TS_PRECT_T850_SHFLX_FLNS_LHFLX_PRECSL_PSL_Q200_Q500_Q850_T200_T500_TAUX_TAUY_TREFHTMX_U010_combined"
            # Remove '_combined' from the end of the varname string and split by '_'
            desired_order = varname.replace("_combined", "").split(",")

            # Get the current 'sample' coordinate, which contains the variable labels
            # filename_order_string = f"{varname}"

            filename_order_string = "FLNS,FLNSC,FSNS,FSNSC,LHFLX,PRECL,PRECSC,PRECSL,PRECT,PSL,Q200,Q500,Q850,QBOT,SHFLX,T010,T200,T500,T850,TAUX,TAUY,TMQ,TREFHT,TREFHTMN,TREFHTMX,TS,U010,U200,U500,U850,UBOT,V200,V500,V850,VBOT,WSPDSRFAV,Z050,Z500,bc_a1_SRF,dst_a1_SRF,dst_a3_SRF,pom_a1_SRF,so4_a1_SRF,so4_a2_SRF,so4_a3_SRF,soa_a1_SRF,soa_a2_SRF"
            current_order = filename_order_string.split(",")

            # Repeat each variable name in `current_order` 100 times to match the data structure
            expanded_current_order = np.repeat(current_order, 1600)

            # Create an index array to reorder the data according to `desired_order`
            reorder_index = np.concatenate([
                np.where(expanded_current_order == var)[0] for var in desired_order if var in expanded_current_order
            ])

            # Reorder the DataArray along the 'sample' dimension
            reordered_data = feat_da.isel(sample=reorder_index)

            print("Data has been reordered according to the specified varname order, continuing.")

            features_list.append(reordered_data.values.flatten())
            continue
        if feature in [
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
            "fftmax"
        ]:
            # let's log the feature and value of i if i is a multiple of 10
            logging.info(f"Computing feature {feature}")

            feat_da = dc.get_calc(feature)
        else:
            # also log here the feature and value of i if i is a multiple of 10
            # if i % 10 == 0:
            logging.info(f"Computing feature {feature}")
            # For features that don't depend on spatial dimensions


            # dc_nospatial = ldcpy.Datasetcalcs(
            #     sample_da, "cam-fv", [], weighted=False
            # )
            # feat_da = dc_nospatial.get_single_calc(feature)


            # Loop over each sample in sample_da, apply get_single_calc, and collect results
            results = []
            for i in range(len(sample_da['sample'])):
                single_sample = sample_da.isel(sample=i)
                dc_nospatial_single = ldcpy.Datasetcalcs(single_sample, "cam-fv", ["lat", "lon"], weighted=False)
                result = dc_nospatial_single.get_single_calc(feature)
                if hasattr(result, 'values'):
                    # If result.values is a single-element array, extract the scalar
                    if isinstance(result.values, np.ndarray) and result.values.size == 1:
                        result = result.values.item()  # Convert single-element arrays to scalars
                    elif isinstance(result.values, np.ndarray):
                        result = result.values[0]  # Or handle multi-element arrays as needed
                else:
                    # result is already a scalar, so no further processing is needed
                    pass
                results.append(result)

            # Convert the list of results to a numpy array, then to a DataArray
            feat_da = xr.DataArray(
                np.array(results),
                dims=["sample"],
                coords={"sample": sample_da["sample"]}
            )

        # save the feature to file
        # Flatten the MultiIndex by resetting it to coordinates
        # flat_da = feat_da.reset_index('sample')  # Replace 'sample' with the appropriate dimension name if different
        flat_da = feat_da

        # Save as a NetCDF file
        if 'weighted' in flat_da.attrs:
            del flat_da.attrs['weighted']

        # if there's a multi-index, we need to handle it
        if 'multi_index' in flat_da.indexes:
            # Now remove the MultiIndex by dropping the dimension
            flat_da = flat_da.drop_vars('multi_index')

        flat_da.to_netcdf(f"{storage_loc}/{varname}_{orig_label}_FEATURE_{feature}_{m}_time{times[0]}_second.nc")

        # Convert feature DataArray to numpy array
        feat_np = feat_da.values

        features_list.append(feat_np)



    # Convert features_list to numpy array
    features_np = np.array(features_list)
    return features_np


if __name__ == '__main__':
    import multiprocessing
    # Set the multiprocessing start method for macOS/Windows
    multiprocessing.set_start_method('spawn', force=True)
    main()