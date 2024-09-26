# main.py
import os
import sys
import json
import numpy as np
import xarray as xr
import ldcpy
from training import train_cnn
from classification_labels import classify

# Read the JSON configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Extract parameters
location = config.get('Location')
save_dir = config.get('SaveDir')
var_list = config.get('VarList')
filename_pre = config.get('FilenamePre')
filename_post = config.get('FilenamePost')
orig_path = config.get('OrigPath')
comp_path = config.get('CompPath')
sub_dirs = config.get('SubDirs')
comp_dirs = config.get('CompDirs')
opt_ldcpy_dev_path = config.get('OptLdcpyDevPath')
times = config.get('Times')
navg = config.get('Navg')
storage_loc = config.get('StorageLoc')
stride = config.get('Stride')
cut_dataset = config.get('CutDataset')
metric = config.get('Metric')

# Add ldcpypath to sys.path
if opt_ldcpy_dev_path:
    sys.path.insert(0, opt_ldcpy_dev_path)

# Initialize lists
files = []
labels = []

# For each variable in var_list
for idx, varname in enumerate(var_list):
    pre = filename_pre[idx]
    post = filename_post[idx]

    # Original data files and labels
    for subdir in sub_dirs:
        orig_file = os.path.join(orig_path, subdir, pre + varname + post)
        files.append(orig_file)
        labels.append('orig_' + subdir)

    # Compressed data files and labels
    for comp_dir in comp_dirs:
        for subdir in sub_dirs:
            comp_file = os.path.join(comp_path, comp_dir, subdir, pre + varname + post)
            files.append(comp_file)
            labels.append(comp_dir + '_' + subdir)

# Open the datasets
data_type = 'cam-fv'

dataset_col = ldcpy.open_datasets(
    list_of_files=files,
    labels=labels,
    data_type=data_type,
    varnames=var_list
)

# Prepare labels for original and compressed datasets
orig_labels = ['orig_' + subdir for subdir in sub_dirs]
comp_labels = [comp_dir + '_' + subdir for comp_dir in comp_dirs for subdir in sub_dirs]

# Initialize dictionaries to store metrics
metrics_data = {m: [] for m in metric}

# Iterate over variables
for varname in var_list:
    # Select the variable data
    var_data_col = dataset_col[varname]

    # Iterate over compressed datasets
    for comp_label in comp_labels:
        # Get the corresponding original dataset (assuming matching subdir)
        subdir = comp_label.split('_')[-1]
        orig_label = 'orig_' + subdir

        # Extract datasets
        ds_orig = var_data_col.sel(collection=orig_label)
        ds_comp = var_data_col.sel(collection=comp_label)

        # Ensure time alignment
        ds_orig, ds_comp = xr.align(ds_orig, ds_comp, join='inner')

        # Compute metrics using ldcpy.Diffcalcs
        dc = ldcpy.Diffcalcs(ds_orig, ds_comp, data_type=data_type)
        for m in metric:
            if m == 'dssim':
                dc.get_diff_calc('ssim_fp')
                dssim_values = dc._ssim_mat_fp[0].values.flatten()
                metrics_data[m].extend(dssim_values)
            elif m == 'pcc':
                pcc_values = dc.get_diff_calc('pearson_correlation_coefficient').values.flatten()
                metrics_data[m].extend(pcc_values)
            elif m == 'spre':
                spre_values = dc.get_diff_calc('spatial_rel_error').values.flatten()
                metrics_data[m].extend(spre_values)

# Generate labels using the classify function
# For simplicity, we can use one of the metrics (e.g., 'dssim') to generate labels
labels_array = np.array(metrics_data['dssim'])

# Generate classification labels
newlabels, metric_dict = classify('config.json', metric, labels_array)

# Prepare the data
# Extract original data
orig_data_list = []
for orig_label in orig_labels:
    ds_orig = dataset_col[var_list].sel(collection=orig_label)
    orig_data_list.append(ds_orig)

# Concatenate data from all original datasets
orig_data_combined = xr.concat(orig_data_list, dim='time')

# Flatten spatial dimensions
data_flat = orig_data_combined.to_array().values.reshape(orig_data_combined.sizes['time'], -1)

# Ensure labels match data
assert data_flat.shape[0] == newlabels.shape[0], "Mismatch between data samples and labels"

# Convert data_flat back to xarray Dataset if required
dataset_xr = xr.Dataset({'combined': (['time', 'space'], data_flat)})

# Call train_cnn
time = data_flat.shape[0]
nvar = len(var_list)

train_cnn(
    dataset=dataset_xr,
    labels=newlabels,
    time=time,
    varname='combined',
    nvar=nvar,
    storageloc=storage_loc,
    metric=metric
)
