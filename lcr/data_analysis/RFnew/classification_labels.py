# first, grab the data from the StorageLoc location,
# saved as xarray dataarrays as train_data_dssim/test_data_dssim and so on.
# load that in for all the metrics (dssim, spre, pcc)


# data is named labels_dssim_RF_local_dssimzfp1.0rf1.npy,labels_dssim_RF_local_dssimzfp1e-1rf1.npy
# and labels_dssim_RF_local_dssimzfp1e-3rf1.npy for the dssim labels, and so on for the other metrics
# using the true thresholds (which should be >0.995 for dssim, <5 for spre, and >0.9995 for pcc),
# come up with
# the label as a compression level for each of the time slices
# then save the labels with the same type as the original dataset


# then we just need to point the data in the model_training function in
# model_training.py to the correct location for the new labels
# we also need to modify the models to work as classifiers instead of regressors
# if the labels it gets are not continuous
# will have to add an option for this

import numpy as np
import xarray as xr
import os
from utils import read_parameters_from_json



def extract_compression_level(filename):
    # Adjusted to extract the compression level from the new filename format
    parts = filename.split('_')
    if len(parts) > 1:
        return parts[1]  # The part after 'zfp' and before the metric name
    return None

def load_and_label_data(metric_info, storageloc, ds=None):

    final_labels_dict = {}

    for metric, info in metric_info.items():
        labels = []  # To hold label DataArrays for each file
        compression_levels = []  # To hold compression levels for each file

        # ds = None
        if ds is None:
            for filename in info['filenames']:
                compression_level = extract_compression_level(filename)
                if compression_level:
                    compression_levels.append(compression_level)

                    filepath = os.path.join(storageloc, filename)
                    data = np.load(filepath)
                    data_da = xr.DataArray(data)

                    if info['comparison'] == 'gt':
                        labels.append(data_da >= info['threshold'])
                    else:  # 'lt'
                        labels.append(data_da < info['threshold'])
        else:
            for key in ds:
                compression_levels.append(key)
                data = ds[key]
                data_da = xr.DataArray(data)

                if info['comparison'] == 'gt':
                    labels.append(data_da >= info['threshold'])
                else:  # 'lt'
                    labels.append(data_da < info['threshold'])



        final_labels = xr.full_like(labels[0], "None", dtype="object")
        for compression_level, label_da in zip(compression_levels, labels):
            final_labels = xr.where((final_labels == "None") & label_da, compression_level, final_labels)

        final_labels_dict[metric] = final_labels

    return final_labels_dict


def compare_across_metrics(final_labels_dict):
    metrics = list(final_labels_dict.keys())
    # Use the first metric as the base for comparison
    base_metric_labels = final_labels_dict[metrics[0]]

    # Initialize the final comparison array with "None" labels
    final_comparison_labels = xr.full_like(base_metric_labels, "None", dtype="object")

    # Iterate through all points and compare labels across metrics
    for metric in metrics:
        current_metric_labels = final_labels_dict[metric]
        final_comparison_labels = xr.where(
            (final_comparison_labels == "None") & (current_metric_labels != "None"),
            current_metric_labels,
            final_comparison_labels
        )
        final_comparison_labels = xr.where(
            (base_metric_labels == current_metric_labels) & (current_metric_labels != "None"),
            current_metric_labels,
            final_comparison_labels
        )

    return final_comparison_labels

def setup(config_path, metrics, cdirs):

    # split config_path to get the directory, filename, and extension
    directory, filename = os.path.split(config_path)
    filenameonly, extension = os.path.splitext(filename)


    # Metrics, their thresholds, and comparison types
    # metrics = ['dssim', 'pcc']
    thresholds = {'dssim': 0.995, 'spre': 5, 'pcc': 0.9995, 'ks': 0.05}
    comparisons = {'dssim': 'gt', 'spre': 'lt', 'pcc': 'gt', 'ks': 'lt'}

    # Base pattern for filenames
    # deal with time, jobid
    filename_pattern = '{level}_{metric}_mat_alltime_20_{filenameonly}1.npy'

    # Generate metrics_info dictionary
    metrics_info = {
        metric: {
            'threshold': thresholds[metric],
            'filenames': [
                filename_pattern.format(metric=metric, level=level, filenameonly=filenameonly) for level in cdirs
            ],
            'comparison': comparisons[metric]
        }
        for metric in metrics
    }

    print(metrics_info)
    return metrics_info

def classify(config_path, metrics, labels=None):
    # config_path = "RF_local_dssim.json"

    save, vlist, pre, post, opath, cpath, cdirs, ldcpypath, time, storageloc, navg, stride, m, cut_dataset, subdirs = read_parameters_from_json(
        config_path)

    # metrics = ['dssim', 'spre', 'pcc', 'ks']
    metrics_info = setup(config_path, metrics, cdirs)

    # Load and label data for each metric
    final_labels_dict = load_and_label_data(metrics_info, storageloc, labels)

    # Compare results across all metrics
    final_comparison_labels = compare_across_metrics(final_labels_dict)

    # Save the final comparison labels
    final_labels_filepath = os.path.join(storageloc, 'final_comparison_labels.npy')
    np.save(final_labels_filepath, final_comparison_labels.values)
    return final_comparison_labels

# Run the function to load, label, and save data
if __name__ == '__main__':
    config_path = "RF_local_spre.json"

    save, vlist, pre, post, opath, cpath, cdirs, ldcpypath, time, storageloc, navg, stride, m, cut_dataset, subdirs = read_parameters_from_json(
        config_path)

    metrics = ['dssim', 'spre', 'pcc', 'ks']
    metrics = ['spre']
    metrics_info, storageloc = setup(config_path, metrics)

    # Load and label data for each metric
    final_labels_dict = load_and_label_data(metrics_info, storageloc)

    # Compare results across all metrics
    final_comparison_labels = compare_across_metrics(final_labels_dict)

    # Save the final comparison labels
    final_labels_filepath = os.path.join(storageloc, 'final_comparison_labels.npy')
    np.save(final_labels_filepath, final_comparison_labels.values)

