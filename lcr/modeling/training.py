import xarray as xr
import os
import numpy as np
import pickle
import datetime
import tensorflow as tf
from sklearn.preprocessing import quantile_transform, LabelEncoder
from sklearn.model_selection import train_test_split
import sklearn
from math import floor
from classification_labels import classify
# os.environ["HDF5_PLUGIN_PATH"]
#
# logging
import logging

import dask
dask.config.set(**{'array.slicing.split_large_chunks': True})

LATS = 182
LONS = 288
WINDOWSIZE = 11

import numpy as np


def xarray_train_test_split(dataset, labels, test_size=0.5, random_state=None):
    # Determine the number of samples
    total_samples = dataset.sizes['sample']

    # Generate shuffled indices
    # if random_state is not None:
    #     np.random.seed(random_state)
    indices = np.arange(total_samples)
    # np.random.shuffle(indices)

    # Determine the size of the test set
    test_size = int(total_samples * test_size)

    # Split indices
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    # Select the train and test sets
    X_train = dataset.isel(sample=train_indices)
    X_test = dataset.isel(sample=test_indices)
    y_train = labels[train_indices]
    y_test = labels[test_indices]

    return X_train, X_test, y_train, y_test


def split_data(
    dataset: xr.Dataset,
    label: np.ndarray,
    time: int,
    nvar: int,
    testset: str,
    lats: int,
    lons: int,
    cut_windows: bool = True,
    window_size: int = 11,
    encoder=None,
    storageloc=None,
    metric=None,
    modeltype=None,
    jobid=None,
    j=None
) -> tuple:
    """
    Splits the data into training, validation, and testing sets based on the `testset` variable.

    If `testset` is set to "1var", all the time steps for a single variable are used as the test set.
    For example:
    - If the dataset has shape (192, 288, 320) and `nvar` is 2, the first 160 samples are used for testing,
      and the remaining 160 samples are split between training and validation.
    - For 3 variables, two-thirds of the data are used for testing, and the remaining third for training and validation.

    Parameters:
    - dataset (xr.Dataset): The dataset containing the data samples.
    - label (np.ndarray): The labels corresponding to the data samples.
    - time (int): Not used in this function (can be removed if not needed).
    - nvar (int): Number of variables in the dataset.
    - testset (str): Strategy for splitting the data. If "1var", uses one variable for testing.
    - lats (int): Number of latitude points.
    - lons (int): Number of longitude points.
    - cut_windows (bool): Whether to cut the data into smaller windows.
    - window_size (int): Size of the window if `cut_windows` is True.
    - encoder: Optional encoder for labels.
    - storageloc: Storage location (not used here).
    - metric: Metric used (not used here).
    - modeltype: Type of model (not used here).
    - jobid: Job ID (not used here).
    - j: Index (not used here).

    Returns:
    - X_train: Training data samples.
    - y_train: Training labels.
    - X_val: Validation data samples.
    - y_val: Validation labels.
    - X_test: Testing data samples.
    - y_test: Testing labels.
    """

    # Total number of samples
    total_samples = dataset.dims['sample']

    if testset == '1var':
        # Calculate the number of samples per variable
        samples_per_var = total_samples // nvar

        # Number of samples to use for testing (all samples from one variable)
        test_samples = samples_per_var

        logging.info(f"Total samples: {total_samples}")
        # Indices for the test set (first variable)
        test_indices = np.arange(0, 2*test_samples)
        # indices for the validation set (second variable)
        # val_indices = np.arange(test_samples, 2*test_samples)

        logging.info(f"Test indices: {test_indices}")
        # Indices for training and validation sets (remaining variables)
        train_val_indices = np.arange(test_samples, total_samples)

        logging.info(f"Train/val indices: {train_val_indices}")
        # Extract test data and labels
        X_test_val = dataset.isel(sample=test_indices)
        y_test_val = label[test_indices]


        # Extract training and validation data and labels
        X_train = dataset.isel(sample=train_val_indices)
        y_train = label[train_val_indices]

        # Further split training and validation sets (e.g., 80% training, 20% validation)
        X_test, X_val, y_test, y_val = xarray_train_test_split(
            X_test_val, y_test_val, test_size=0.5, random_state=None
        )

    else:
        # Default random split if testset is not '1var'
        X_train_val, X_test, y_train_val, y_test = xarray_train_test_split(
            dataset, label, test_size=0.2, random_state=42
        )
        # Further split training and validation sets
        X_train, X_val, y_train, y_val = xarray_train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=42
        )
        # This results in 60% train, 20% val, 20% test

    # Optionally cut the data into smaller windows
    if cut_windows:
        def cut_into_windows(data):
            # Assuming data has dimensions (sample, lat, lon, variable)
            data_windows = []
            for i in range(data.dims['sample']):
                sample = data.isel(sample=i)
                # Convert to numpy array
                sample_array = sample['combined'].values  # Shape: (lat, lon, variable)
                # Pad the sample if necessary
                pad_size = window_size // 2
                sample_padded = np.pad(
                    sample_array,
                    pad_width=((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
                    mode='reflect'
                )
                # Extract windows
                lat_indices = range(pad_size, pad_size + lats)
                lon_indices = range(pad_size, pad_size + lons)
                for lat in lat_indices:
                    for lon in lon_indices:
                        window = sample_padded[
                            lat - pad_size: lat + pad_size + 1,
                            lon - pad_size: lon + pad_size + 1,
                            :
                        ]
                        data_windows.append(window)
            return np.array(data_windows)

        # Cut data into windows
        X_train = cut_into_windows(X_train)
        X_val = cut_into_windows(X_val)
        X_test = cut_into_windows(X_test)

    else:
        # Convert data to numpy arrays
        # X_train = X_train['combined'].values
        # X_val = X_val['combined'].values
        # X_test = X_test['combined'].values
        pass

    # Ensure labels are numpy arrays
    # y_train = np.array(y_train)
    # y_val = np.array(y_val)
    # y_test = np.array(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test

def split_data_old(dataset: xr.Dataset, label: np.ndarray, time: int, nvar: int, testset: str, lats:int, lons:int, cut_windows:bool = True, window_size: int = 11, encoder=None, storageloc=None, metric=None, modeltype=None, jobid=None, j=None) -> tuple:
    """
    Splits the dataset into training, validation, and testing sets based on the specified testset parameter.

    Parameters:
     dataset (xr.Dataset): The input xarray dataset.
     dssim (np.ndarray): The SSIM matrix corresponding to the dataset.
     time (int): The number of time slices to consider in the dataset.
     nvar (int): The number of variables to consider in the dataset.
     testset (str): The method of splitting the data ("random", "oneout", "10pct", "1var", "60pct", "60_25_wholeslice").
     comp (int): The component index to use for extracting labels from the SSIM matrix.

    Returns:
     tuple: A tuple containing the training, validation, and testing data and labels.
    """
    # Calculate the number of windows based on the dataset's dimensions
    # num_windows = lats * lons # 182 * 288
    num_windows = 1

    total_data_points = num_windows * time * nvar
    # Calculate the indices corresponding to 60% and 75% of the data
    index_60pct = int(total_data_points * 0.6)
    index_75pct = int(total_data_points * 0.75)
    index_10pct = int(total_data_points * 0.1)
    index_50pct = int(total_data_points * 0.5)

    # use 90% of the data for training, 9% for validation, and 1% for testing
    if testset == "random":
        train_data, test_data, train_labels, test_labels = train_test_split(dataset[0:(num_windows * time * nvar)],
                                                                            label, test_size=0.5)
        val_data, test_data, val_labels, test_labels = train_test_split(test_data, test_labels, test_size=0.5)
    elif testset == "oneout":
        # train_data = dataset[0:(50596*time*nvar-1)]
        # train_labels = label[0:(50596*time*nvar-1)]
        val_data = dataset[(num_windows * time * nvar - 2):(num_windows * time * nvar - 1)]
        val_labels = label[(num_windows * time * nvar - 2):(num_windows * time * nvar - 1)]
        # test_data = dataset[(50596*time*nvar-1):(50596*time*nvar)]
        # test_labels = label[(50596*time*nvar-1):(50596*time*nvar)]

        # Currently, this will always make the first time slice of the data the test set for consistency
        test_data = dataset[0:num_windows]
        test_labels = label[0:num_windows]
        train_data = dataset[num_windows:(num_windows * time * nvar)]
        train_labels = label[num_windows:(num_windows * time * nvar)]
    elif testset == "10pct":
        # use the first 10% of the data for testing
        test_data = dataset[0:int(num_windows * time * nvar * 0.1)]
        test_labels = label[0:int(num_windows * time * nvar * 0.1)]
        # Note: this randomizes the training and validation data, an alternative would be to use the last 10% of the data for validation
        train_data, val_data, train_labels, val_labels = train_test_split(
            dataset[int(num_windows * time * nvar * 0.1):(num_windows * time * nvar)],
            label[int(num_windows * time * nvar * 0.1):(num_windows * time * nvar)], test_size=0.1)

    elif testset == "1var":
        # leave out a single variable for testing, and use the rest for training and validation
        test_data = dataset[0:(num_windows * time)]
        if label is not None:
            test_labels = label[0:(num_windows * time)]

        # This will randomize the training and validation data, an alternative would be to use the last variable(s) for validation
        if label is not None:
            train_data, val_data, train_labels, val_labels = train_test_split(dataset[(num_windows * time):(num_windows * time * nvar)],
                                                                              label[
                                                                              (num_windows * time):(num_windows * time * nvar)],
                                                                              test_size=0.1)
        else:
            train_data, val_data = train_test_split(dataset[(num_windows * time):(num_windows * time * nvar)],
                                                                              test_size=0.1)

        # Alternatively, use the last variable(s) for validation
        # val_data = dataset[(50596*time*(nvar-1)):(50596*time*nvar)]
        # val_labels = label[(50596*time*(nvar-1)):(50596*time*nvar)]
        # train_data = dataset[(50596*time):(50596*time*(nvar-1))]
        # train_labels = label[(50596*time):(50596*time*(nvar-1))]
    elif testset == "60pct":
        # Use the first 60% of the data for training
        train_data = dataset[0:index_60pct]
        train_labels = label[0:index_60pct]

        # Use the last 25% of the data for testing and validation
        last_25pct_data = dataset[index_75pct:]
        last_25pct_labels = label[index_75pct:]

        # Randomly split the last 25% into 10% for testing and 15% for validation
        test_data, val_data, test_labels, val_labels = train_test_split(
            last_25pct_data, last_25pct_labels, test_size=0.4)  # 10% of 25% is 40% of the last 25%

    elif testset == "60_25_wholeslice":
        # Calculate the number of time slices in the last 25% of the data (rounding down)
        num_time_slices_last_25pct = (total_data_points - index_75pct) // num_windows

        # Calculate the number of time slices for test and validation (rounding down)
        num_time_slices_val = num_time_slices_last_25pct * 40 // 100.0
        num_time_slices_test = num_time_slices_last_25pct - num_time_slices_val

        # Calculate the number of windows for test and validation
        num_windows_test = num_time_slices_test * num_windows
        num_windows_val = num_time_slices_val * num_windows

        # Calculate the start index for the remaining time slice (if any)
        remaining_start_index = index_75pct + num_windows_test + num_windows_val

        # Use the first 60% of the data for training
        train_data = dataset[0:index_60pct]
        train_labels = label[0:index_60pct]

        # Use the calculated number of windows for test and validation
        test_data = dataset[index_75pct:int(index_75pct + num_windows_test)]
        test_labels = label[index_75pct:int(index_75pct + num_windows_test)]
        val_data = dataset[int(index_75pct + num_windows_test):int(index_75pct + num_windows_test + num_windows_val)]
        val_labels = label[int(index_75pct + num_windows_test):int(index_75pct + num_windows_test + num_windows_val)]

        # If there is a remaining time slice, split it between test and validation to preserve the 60-40 split
        if remaining_start_index < total_data_points:
            remaining_windows = total_data_points - remaining_start_index
            split_index = remaining_windows * 40 // 100
            test_data = np.concatenate(
                (test_data, dataset[int(remaining_start_index):int(remaining_start_index + split_index)]))
            test_labels = np.concatenate(
                (test_labels, label[int(remaining_start_index):int(remaining_start_index + split_index)]))
            val_data = np.concatenate((val_data, dataset[int(remaining_start_index + split_index):]))
            val_labels = np.concatenate((val_labels, label[int(remaining_start_index + split_index):]))


    elif testset == "10_90_wholeslice":
        # Calculate the number of time slices in the last 90% of the data (rounding down)
        num_time_slices_last_90pct = (total_data_points - index_10pct) // num_windows

        # Calculate the number of time slices for test and validation (rounding down)


        num_time_slices_val = num_time_slices_last_90pct // 9
        num_time_slices_test = num_time_slices_last_90pct - num_time_slices_val
        num_time_slices_train = time*nvar - num_time_slices_last_90pct

        # Calculate the number of windows for test and validation
        num_windows_test = num_time_slices_test * num_windows
        num_windows_val = num_time_slices_val * num_windows

        # Calculate the start index for the remaining time slice (if any)
        remaining_start_index = index_10pct + num_windows_test + num_windows_val

        # Use the first 10% of the data for training
        train_data = dataset[0:index_10pct]
        if label is not None:
            train_labels = label[0:index_10pct]

        # Use the calculated number of windows for test and validation
        test_data = dataset[index_10pct:int(index_10pct + num_windows_test)]
        if label is not None:
            test_labels = label[index_10pct:int(index_10pct + num_windows_test)]
        val_data = dataset[int(index_10pct + num_windows_test):int(index_10pct + num_windows_test + num_windows_val)]
        if label is not None:
            val_labels = label[int(index_10pct + num_windows_test):int(index_10pct + num_windows_test + num_windows_val)]

        # If there is a remaining time slice, split it between test and validation to preserve the 60-40 split
        if remaining_start_index < total_data_points:
            remaining_windows = total_data_points - remaining_start_index
            split_index = remaining_windows * 90 // 100
            test_data = np.concatenate(
                (test_data, dataset[int(remaining_start_index):int(remaining_start_index + split_index)]))
            if label is not None:
                test_labels = np.concatenate(
                    (test_labels, label[int(remaining_start_index):int(remaining_start_index + split_index)]))
            val_data = np.concatenate((val_data, dataset[int(remaining_start_index + split_index):]))
            if label is not None:
                val_labels = np.concatenate((val_labels, label[int(remaining_start_index + split_index):]))

        if not cut_windows:
            train_data = dataset[0:int(index_10pct/num_windows)]
            if label is not None:
                train_labels = label[0:int(index_10pct/num_windows)]
            val_data = dataset[int(index_10pct/num_windows):(int(index_10pct/num_windows)+int(num_windows_val/num_windows))]
            if label is not None:
                val_labels = label[int(index_10pct/num_windows):(int(index_10pct/num_windows)+int(num_windows_val/num_windows))]
            test_data = dataset[(int(index_10pct/num_windows)+int(num_windows_val/num_windows)):(int(index_10pct/num_windows)+int(num_windows_val/num_windows)+int(num_windows_test/num_windows))]
            if label is not None:
                test_labels = label[(int(index_10pct/num_windows)+int(num_windows_val/num_windows)):(int(index_10pct/num_windows)+int(num_windows_val/num_windows)+int(num_windows_test/num_windows))]

    elif testset == "50_50_wholeslice":

        # Calculate the number of time slices in the last 90% of the data (rounding down)
        num_time_slices_last_50pct = (total_data_points - index_50pct) // num_windows

        # Calculate the number of time slices for test and validation (rounding down)


        num_time_slices_val = num_time_slices_last_50pct // 2
        num_time_slices_test = num_time_slices_last_50pct - num_time_slices_val
        num_time_slices_train = time*nvar - num_time_slices_last_50pct

        # Calculate the number of windows for test and validation
        num_windows_test = num_time_slices_test * num_windows
        num_windows_val = num_time_slices_val * num_windows

        # Calculate the start index for the remaining time slice (if any)
        remaining_start_index = index_50pct + num_windows_test + num_windows_val
        if cut_windows:
            # Use the first 10% of the data for training
            train_data = dataset[0:index_50pct]
            if label is not None:
                train_labels = label[0:index_50pct]

            # Use the calculated number of windows for test and validation
            test_data = dataset[index_50pct:int(index_50pct + num_windows_test)]
            if label is not None:
                test_labels = label[index_50pct:int(index_50pct + num_windows_test)]
            val_data = dataset[int(index_50pct + num_windows_test):int(index_50pct + num_windows_test + num_windows_val)]
            if label is not None:
                val_labels = label[int(index_50pct + num_windows_test):int(index_50pct + num_windows_test + num_windows_val)]

            # If there is a remaining time slice, split it between test and validation to preserve the 60-40 split
            if remaining_start_index < total_data_points:
                remaining_windows = total_data_points - remaining_start_index
                split_index = remaining_windows * 50 // 100
                test_data = np.concatenate(
                    (test_data, dataset[int(remaining_start_index):int(remaining_start_index + split_index)]))
                if label is not None:
                    test_labels = np.concatenate(
                        (test_labels, label[int(remaining_start_index):int(remaining_start_index + split_index)]))
                val_data = np.concatenate((val_data, dataset[int(remaining_start_index + split_index):]))
                if label is not None:
                    val_labels = np.concatenate((val_labels, label[int(remaining_start_index + split_index):]))
        elif not cut_windows:
            train_data = dataset[0:int(index_50pct/num_windows)]
            if label is not None:
                train_labels = label[0:int(index_50pct/num_windows)]
                # use the encoder to transform the labels back to the original labels
                original_labels = encoder.inverse_transform(train_labels)
                # compute the number of labels in each class
                unique, counts = np.unique(original_labels, return_counts=True)
                # save the number of labels in each class as a text file
                with open(f"{storageloc}train_labels_{metric}_{j}{time}{modeltype}_{jobid}.txt", "w") as f:
                    for i in range(len(unique)):
                        f.write(f"{unique[i]}: {counts[i]}\n")
            val_data = dataset[int(index_50pct/num_windows):(int(index_50pct/num_windows)+int(num_windows_val/num_windows))]
            if label is not None:
                val_labels = label[int(index_50pct/num_windows):(int(index_50pct/num_windows)+int(num_windows_val/num_windows))]
            test_data = dataset[(int(index_50pct/num_windows)+int(num_windows_val/num_windows)):(int(index_50pct/num_windows)+int(num_windows_val/num_windows)+int(num_windows_test/num_windows))]
            if label is not None:
                test_labels = label[(int(index_50pct/num_windows)+int(num_windows_val/num_windows)):(int(index_50pct/num_windows)+int(num_windows_val/num_windows)+int(num_windows_test/num_windows))]

    if label is not None:
        return train_data, train_labels, val_data, val_labels, test_data, test_labels
    else:
        return train_data, val_data, test_data



def convert_np_to_xr(np_arrays, titles=None):
    das = []
    set_values = [f'set{i+1}' for i in range(len(np_arrays))]
    if titles:
        set_values = titles
    # stick the dssims and predictions into an xarray DataArray
    da = xr.DataArray(np_arrays)
    lat_values = np.linspace(-90, 90, np_arrays.shape[1])
    lon_values = np.linspace(0, 360, np_arrays.shape[2])
    len_values = np.arange(np_arrays.shape[0])
    # Rotate the longitude values by 180 degrees
    lon_values = np.where(lon_values > 180, lon_values - 360, lon_values)
    da.coords['latitude'] = (('dim_1'), lat_values)
    da.coords['longitude'] = (('dim_2'), lon_values)
    # da = da.expand_dims({'length': len_values})
    # Convert the DataArray to a Dataset
    da = da.rename({'dim_0': 'time', 'dim_1': 'latitude', 'dim_2': 'longitude'})
    # Assuming dssims_da is your DataArray and you've renamed the dimensions to 'latitude' and 'longitude'
    da['latitude'].attrs['units'] = 'degrees_north'
    da['longitude'].attrs['units'] = 'degrees_east'
    da.attrs['units'] = ''
    # ds = xr.concat(das, dim=xr.DataArray(set_values, dims='collection'))
    ldcpy_da = da.to_dataset(name='array')
    ldcpy_da.attrs['data_type'] = 'cam-fv'
    ldcpy_da.attrs["dtype"] = np.float64
    return ldcpy_da

def get_data_labels(dataset: xr.Dataset, labels: np.ndarray, time, varname, nvar, storageloc,
                                   testset="random", j=None, plotdir=None, window_size=WINDOWSIZE, only_data=False, modeltype="cnn", feature=None, featurelist=None, transform="quantile", jobid=0, cut_windows=True, metric=["dssim"]) -> float:
    """
    Train a CNN for DSSIM regression and return the average error.

    Parameters:
        dataset (xr.Dataset): The input dataset.
        dssim (np.ndarray): The DSSIM values.
        time (int): The time parameter.
        varname (str): The variable name.
        nvar (int): The number of variables.
        storageloc (str): The storage location.
        testset (str, optional): The test set type. Defaults to "random".
        j (int, optional): An optional parameter. Defaults to None.
        plotdir (str, optional): The plot directory. Defaults to None.

    Returns:
        float: The average error.
    """
    # model_path = f"model_{j}{comp}.h5"
    # model_path = ""
    average_error_path = os.path.join(storageloc, "average_error.txt")

    if not only_data:
        # newlabels, metric_dict = classify(f"{j}.json", metric, labels)

        # save the metric_dict
        # with open(f"{storageloc}metric_dict_{j}{time}{modeltype}{jobid}.pkl", "wb") as f:
        #     pickle.dump(metric_dict, f)


        # get the label keys
        # label_keys = list(labels.keys())

        #### SECTIONS TO REVIEW ####

        label_encoder = LabelEncoder()
        integer_encoded_labels = label_encoder.fit_transform(labels)

    # let's go backwards to the original labels
    scores = []
    av_preds = []
    av_dssims = []
    # for comp in labels.keys():
    #     model_path = f"model_{metric}{j}{comp}{time}{modeltype}{jobid}.h5"
    #     if os.path.exists(model_path):
    #         model = tf.keras.models.load_model(model_path)
    #         with open(f"{storageloc}av_preds_{metric}_{j}{comp}{time}{modeltype}{jobid}", "rb") as f:
    #             av_preds = pickle.load(f)
    #         with open(f"{storageloc}av_{metric}_{j}{comp}{time}{modeltype}{jobid}", "rb") as f:
    #             av_dssims = pickle.load(f)
    #         with open(f"{storageloc}predictions_{metric}_{j}{comp}{time}{modeltype}{jobid}.npy", "rb") as f:
    #             predictions = np.load(f)
    #         with open(f"{storageloc}test_plot_{j}{comp}{time}{modeltype}{jobid}.npy", "rb") as f:
    #             test_plot = np.load(f, allow_pickle=True)
    #         if modeltype == "cnn":
    #             with open(f"{storageloc}scores_{j}{comp}{time}{modeltype}{jobid}", "rb") as f:
    #                 scores = pickle.load(f)


    #### SECTIONS TO REVIEW ^^^^^^^^ ####

    # conver the data to a numpy array

    # for i in range(time*nvar):
        # dataset[i] = (dataset[i] - np.mean(dataset[i])) / np.std(dataset[i])
        # convert newlabels to a numpy array
        # newlabels = np.array(newlabels)
    if not only_data:
        newlabels = np.array(integer_encoded_labels)
        train_data, train_labels, val_data, val_labels, test_data, test_labels = split_data(dataset, newlabels,
                                                                                            time, nvar, testset,
                                                                                                LATS, LONS, cut_windows,
                                                                                                encoder=label_encoder,
                                                                                                storageloc=storageloc,
                                                                                                metric=metric,
                                                                                                modeltype=modeltype,
                                                                                                jobid=jobid, j=j)
    else:
        train_data, val_data, test_data = split_data(dataset, None, time, nvar, testset, LATS, LONS, cut_windows, encoder=None, storageloc=storageloc, metric=metric, modeltype=modeltype, jobid=jobid, j=j)



    # check the echosave directory, open trial_results.csv
    # read the column mean_squared_error to find the row with the minimum value
    # then set filter1, filter2, and dropout to the values in that row
    # csv_path = os.path.join("echosave/trial_results.csv")
    # if os.path.exists(csv_path):
    #     with open(csv_path, "r") as f:
    #         lines = f.readlines()
    #         mse = []
    #         # remember to skip the first line
    #         # for line in lines:
    #         #     mse.append(float(line.split(",")[1]))
    #         for line in lines[1:]:
    #             mse.append(float(line.split(",")[8]))
    #         min_mse = min(mse)
    #         min_mse_index = mse.index(min_mse) + 1
    #         filter1 = int(lines[min_mse_index].split(",")[3])
    #         filter2 = int(lines[min_mse_index].split(",")[4])
    #         dropout = float(lines[min_mse_index].split(",")[6])
    #         batch_size = int(lines[min_mse_index].split(",")[5])
    #         conv_layers = int(lines[min_mse_index].split(",")[7])
    # else:
        # First, convert the train_data to a NumPy array with shape (samples, lat, lon, variables)
    train_data_np = train_data['combined'].transpose('sample', 'lat', 'lon').values
    val_data_np = val_data['combined'].transpose('sample', 'lat', 'lon').values
    test_data_np = test_data['combined'].transpose('sample', 'lat', 'lon').values

    # Adjust the labels if necessary (ensure they are integers starting from 0)
    train_labels_np = train_labels.astype(int)
    val_labels_np = val_labels.astype(int)
    test_labels_np = test_labels.astype(int)

    # save the data and labels as numpy arrays
    np.save(f"{storageloc}/train_data_{j}{time}{modeltype}{jobid}.npy", train_data_np)
    np.save(f"{storageloc}/val_data_{j}{time}{modeltype}{jobid}.npy", val_data_np)
    np.save(f"{storageloc}/test_data_{j}{time}{modeltype}{jobid}.npy", test_data_np)
    np.save(f"{storageloc}/train_labels_{j}{time}{modeltype}{jobid}.npy", train_labels_np)
    np.save(f"{storageloc}/val_labels_{j}{time}{modeltype}{jobid}.npy", val_labels_np)
    np.save(f"{storageloc}/test_labels_{j}{time}{modeltype}{jobid}.npy", test_labels_np)
    # save the encoding also
    with open(f"{storageloc}/label_encoder_{j}{time}{modeltype}{jobid}.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    return

def train_cnn(
        train_data_np,
        train_labels_np,
        val_data_np,
        val_labels_np,
        test_data_np,
        test_labels_np,
        modeltype,
        transform,
        featurelist=None,
):
    """
    Train a model for the given data and labels.

    Parameters:
        train_data_np (np.ndarray): The training data.
        train_labels_np (np.ndarray): The training labels.
        val_data_np (np.ndarray): The validation data.
        val_labels_np (np.ndarray): The validation labels.
        test_data_np (np.ndarray): The testing data.
        test_labels_np (np.ndarray): The testing labels.
        modeltype (str): The model type ("cnn", "rf", "mlp").
        transform (str): The transformation type (e.g., "quantile").
        featurelist (list of str, optional): List of features to compute using ldcpy.

    Returns:
        Trained model.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    # Apply transformation if specified
    if transform == "quantile":
        # Flatten the data for quantile_transform
        n_samples_train = train_data_np.shape[0]
        n_samples_val = val_data_np.shape[0]
        n_samples_test = test_data_np.shape[0]

        train_data_reshaped = train_data_np.reshape(n_samples_train, -1)
        val_data_reshaped = val_data_np.reshape(n_samples_val, -1)
        test_data_reshaped = test_data_np.reshape(n_samples_test, -1)

        # Apply the quantile transform
        train_data_transformed = quantile_transform(
            train_data_reshaped,
            output_distribution="uniform",
            copy=True,
            n_quantiles=1000,
        )
        val_data_transformed = quantile_transform(
            val_data_reshaped,
            output_distribution="uniform",
            copy=True,
            n_quantiles=1000,
        )
        test_data_transformed = quantile_transform(
            test_data_reshaped,
            output_distribution="uniform",
            copy=True,
            n_quantiles=1000,
        )

        # Reshape back to original shape
        train_data_np = train_data_transformed.reshape(train_data_np.shape)
        val_data_np = val_data_transformed.reshape(val_data_np.shape)
        test_data_np = test_data_transformed.reshape(test_data_np.shape)

    # Compute features using ldcpy if featurelist is provided

    # Now, depending on modeltype, build and train the model
    # Define filter sizes, dropout rate, and number of convolutional layers as needed
    filter1 = 32
    filter2 = 64
    dropout = 0.5
    conv_layers = 2

    if modeltype == "cnn":
        # Update input_shape
        input_shape = train_data_np.shape[1:]  # Now input_shape is (192, 288, 1)
        i = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Conv2D(filter1, kernel_size=(3, 3), activation="relu")(i)
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3))(x)
        x = tf.keras.layers.Conv2D(filter2, kernel_size=(3, 3), activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3))(x)
        x = tf.keras.layers.Conv2D(filter2, kernel_size=(3, 3), activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3))(x)
        if conv_layers >= 3:
            x = tf.keras.layers.Conv2D(filter1, kernel_size=(2, 2), activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Conv2D(filter2, kernel_size=(2, 2), activation="relu")(x)
        if conv_layers == 4:
            x = tf.keras.layers.Conv2D(filter2, kernel_size=(2, 2), activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        # softmax for multi-class classification, (number of classes will be the number of unique training validation labels)
        outputs = tf.keras.layers.Dense(len(np.unique(train_labels_np)), activation="softmax")(x)

        model = tf.keras.Model(inputs=i, outputs=outputs)
        # Compile the model for classification
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=["accuracy"],
        )

        model.summary()

        # Train the model
        num_epochs = 10  # Adjust as needed
        batch_size = 5  # Adjust as needed

        model.fit(
            train_data_np,
            train_labels_np,
            validation_data=(val_data_np, val_labels_np),
            epochs=num_epochs,
            batch_size=batch_size,
        )

        # Evaluate the model on test data
        test_loss, test_accuracy = model.evaluate(test_data_np, test_labels_np)
        print(f"Test Accuracy: {test_accuracy}")

        return model

def evaluate_model(model, test_data, test_labels):
    """
    Evaluate the trained model on test data.

    Parameters:
        model: The trained model.
        test_data (np.ndarray): The test data.
        test_labels (np.ndarray): The true labels for the test data.

    Returns:
        accuracy (float): The accuracy of the model on the test data.
        conf_matrix (np.ndarray): The confusion matrix.
        class_report (str): The classification report.
    """
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    # Get predicted probabilities
    predictions_prob = model.predict(test_data)

    # Convert probabilities to class labels
    predictions = np.argmax(predictions_prob, axis=1)

    # Compute accuracy
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Compute confusion matrix
    conf_matrix = confusion_matrix(test_labels, predictions)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Compute classification report
    class_report = classification_report(test_labels, predictions)
    print("Classification Report:")
    print(class_report)

    return accuracy, conf_matrix, class_report

    # # Define filter sizes, dropout rate, and number of convolutional layers as needed
    # filter1 = 32  # Example value, adjust as needed
    # filter2 = 64  # Example value, adjust as needed
    # dropout = 0.5  # Example value, adjust as needed
    # conv_layers = 2  # Adjust as needed (2, 3, or 4)
    #
    # if modeltype == "cnn":
    #     # Assuming train_data_np is of shape (samples, lat, lon, variables)
    #     input_shape = (
    #         np.shape(train_data_np)[1],
    #         np.shape(train_data_np)[2],
    #         np.shape(train_data_np)[3],
    #     )
    #     i = tf.keras.Input(shape=input_shape)
    #     x = tf.keras.layers.Conv2D(filter1, kernel_size=(2, 2), activation="relu")(i)
    #     # If conv_layers is 3 or 4, add another conv layer
    #     if conv_layers == 3 or conv_layers == 4:
    #         x = tf.keras.layers.Conv2D(filter1, kernel_size=(2, 2), activation="relu")(x)
    #     x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    #     x = tf.keras.layers.Conv2D(filter2, kernel_size=(2, 2), activation="relu")(x)
    #     if conv_layers == 4:
    #         x = tf.keras.layers.Conv2D(filter2, kernel_size=(2, 2), activation="relu")(x)
    #     x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    #     x = tf.keras.layers.Flatten()(x)
    #     x = tf.keras.layers.Dropout(dropout)(x)
    #     # This is a binary classification problem, so the output layer has 2 nodes and uses softmax
    #     outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    #
    #     model = tf.keras.Model(inputs=i, outputs=outputs)
    #     # Compile the model for classification
    #     model.compile(
    #         loss="sparse_categorical_crossentropy",
    #         optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    #         metrics=["accuracy"],
    #     )
    #
    #     model.summary()
    #
    # elif modeltype == "rf":
    #     import ldcpy
    #     # fit the model using some standard parameters
    #     model = sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
    #
    # # apply a quantile transformation to each of the lat/lon slices of the data
    # # this will make the data more Gaussian
    # # first flatten the data
    # savelats = train_data_np.shape[1]
    # savelons = train_data_np.shape[2]
    # # Reshape the data to 2D, treating each spatial point as a separate sample
    # train_data_reshaped = train_data_np.reshape(-1, train_data_np.shape[-1])
    # val_data_reshaped = val_data_np.reshape(-1, val_data_np.shape[-1])
    # test_data_reshaped = test_data_np.reshape(-1, test_data_np.shape[-1])
    # test_data_OLD = test_data_np.reshape(-1, test_data_np.shape[-1])
    #
    # # Apply the quantile transform
    # if transform == "quantile":
    #     train_data_transformed = quantile_transform(train_data_reshaped, output_distribution='uniform', copy=True,
    #                                                 n_quantiles=1000)
    #     val_data_transformed = quantile_transform(val_data_reshaped, output_distribution='uniform', copy=True,
    #                                               n_quantiles=1000)
    #     test_data_transformed = quantile_transform(test_data_reshaped, output_distribution='uniform', copy=True,
    #                                                n_quantiles=1000)
    # else:
    #     train_data_transformed = train_data_reshaped
    #     val_data_transformed = val_data_reshaped
    #     test_data_transformed = test_data_reshaped
    #
    # # Reshape the data back to its original 3D shape
    # train_data = train_data_transformed.reshape(-1, savelats, savelons)
    # val_data = val_data_transformed.reshape(-1, savelats, savelons)
    # test_data = test_data_transformed.reshape(-1, savelats, savelons)
    #
    # test_data_OLD = test_data_OLD.reshape(-1, savelats, savelons)
    #
    #
    # if modeltype == "rf":
    #     if feature is not None:
    #         # train_data_xr = convert_np_to_xr(train_data)
    #         # dc = ldcpy.Datasetcalcs(train_data_xr.to_array(), train_data_xr.data_type, ["latitude", "longitude"], weighted=False)
    #         # ns = dc.get_calc("ns_con_var")
    #         # ew = dc.get_calc("ew_con_var")
    #         # we_fd = dc.get_calc("w_e_first_differences")
    #         # ns_fd = dc.get_calc("n_s_first_differences")
    #         # fftr = dc.get_calc("fftratio")
    #         # mg = dc.get_calc("magnitude_range")
    #         # npns = ns.to_numpy()
    #         # npew = ew.to_numpy()
    #         # npwe_fd = we_fd.to_numpy()
    #         # npns_fd = ns_fd.to_numpy()
    #         # np_fftr = fftr.to_numpy()
    #         # np_mg = mg.to_numpy()
    #         # train_data = np.concatenate((npns, npew, npwe_fd, npns_fd, np_fftr, np_mg.reshape(1, np_mg.shape[0])), axis=0)
    #         # train_data = train_data.transpose()
    #
    #         for type in ["train", "test"]:
    #             if type == "test":
    #                 test_data_xr = convert_np_to_xr(test_data)
    #                 # squeeze the data
    #                 test_data_xr = test_data_xr.squeeze()
    #                 dc = ldcpy.Datasetcalcs(test_data_xr.to_array(), test_data_xr.data_type, ["latitude", "longitude"], weighted=False)
    #             elif type == "train":
    #                 train_data_xr = convert_np_to_xr(train_data)
    #                 # squeeze the data
    #                 train_data_xr = train_data_xr.squeeze()
    #                 dc = ldcpy.Datasetcalcs(train_data_xr.to_array(), train_data_xr.data_type, ["latitude", "longitude"], weighted=False)
    #             if feature in ["ns_con_var", "ew_con_var", "w_e_first_differences", "n_s_first_differences", "fftratio", "fftmax", 'w_e_first_differences_max', 'n_s_first_differences_max', 'mean']:
    #                 ns = dc.get_calc(feature)
    #             else:
    #
    #                 dc = ldcpy.Datasetcalcs(train_data_xr.to_array(), train_data_xr.data_type, [], weighted=False)
    #                 ns = dc.get_single_calc(feature)
    #             npns = ns.to_numpy()
    #             # save train_data, train_labels, val_data, val_labels, test_data, test_labels
    #
    #             np.save(f"{storageloc}{feature}_{metric}_{type}{time}{jobid}_classify.npy", npns)
    #         exit(0)
    #
    #     for type in ["train", "test"]:
    #         flist = None
    #         for f in featurelist:
    #             if flist is None and f != "magnitude_range":
    #                 flist = np.load(f"{storageloc}{f}_{metric}_{type}{time}{jobid}_classify.npy")
    #             elif flist is None and f == "magnitude_range":
    #                 flist = np.load(f"{storageloc}{f}_{metric}_{type}{time}{jobid}_classify.npy")
    #                 flist = flist.reshape(1, flist.shape[0])
    #             elif f == "magnitude_range":
    #                 feat = np.load(f"{storageloc}{f}_{metric}_{type}{time}{jobid}_classify.npy")
    #                 flist = np.concatenate((flist, feat.reshape(1, feat.shape[0])), axis=0)
    #             else:
    #                 feat = np.load(f"{storageloc}{f}_{metric}_{type}{time}{jobid}_classify.npy")
    #                 flist = np.concatenate((flist, feat), axis=0)
    #             if type == "train":
    #                 train_data = flist
    #             elif type == "test":
    #                 test_data = flist
    #
    #     train_data = train_data.transpose()
    #     test_data = test_data.transpose()
    #
    #
    # # save train_data, train_labels, val_data, val_labels, test_data, test_labels
    # np.save(f"{storageloc}train_data_CNN11_local_classify.npy", train_data)
    # np.save(f"{storageloc}train_labels_{metric}_CNN11_local_classify.npy", train_labels)
    # np.save(f"{storageloc}val_data_CNN11_local_classify.npy", val_data)
    # np.save(f"{storageloc}val_labels_{metric}_CNN11_local_classify.npy", val_labels)
    # np.save(f"{storageloc}test_data_CNN11_local_classify.npy", test_data)
    # np.save(f"{storageloc}test_labels_{metric}_CNN11_local_classify.npy", test_labels)
    #
    # if only_data:
    #     exit()
    #
    # #plot the first (LATS * LONS) values of test_labels in a 182*288 grid
    # # make this the first of a 1x4 subplot
    #     # fit the model
    # if modeltype == "cnn":
    #     history = model.fit(train_data, train_labels, epochs=6, batch_size=batch_size, validation_data=(val_data, val_labels))
    #     score = model.evaluate(test_data, verbose=0)
    #     print('Test loss:', score[0])
    #     print('Test accuracy:', score[1])
    #     # let's save the training history
    #     with open(f"{storageloc}cnn_train_history_{j}{time}{jobid}{modeltype}_classify", "wb") as f:
    #         pickle.dump(history.history, f)
    # elif modeltype == "rf":
    #     train_data[train_data == -np.inf] = 0
    #     train_data[train_data == np.inf] = 0
    #     train_data[train_data == np.nan] = 0
    #     test_data[test_data == -np.inf] = 0
    #     test_data[test_data == np.inf] = 0
    #     train_data[train_data == np.nan] = 0
    #     val_data[val_data == -np.inf] = 0
    #     val_data[val_data == np.inf] = 0
    #     train_data[np.isnan(train_data)] = 0
    #     test_data[np.isnan(test_data)] = 0
    #     val_data[np.isnan(val_data)] = 0
    #     model.fit(train_data, train_labels)
    #     # get the feature importances and save them
    #
    # val_predictions = model.predict(test_data)
    # if modeltype == "cnn":
    #     predictions = np.argmax(val_predictions, axis=1)
    # else:
    #     predictions = model.predict(test_data)
    # label_predictions = label_encoder.inverse_transform(predictions)
    # if modeltype == "rf":
    #     # save the feature importances as {storageloc}importances_{i}_{j.split('.')[0]}{jobid}{model}
    #     importances = model.feature_importances_
    #     np.save(f"{storageloc}importances_{j}{time}{jobid}{modeltype}_classify.npy", importances)
    #     # convert the importances to text and save them
    #     with open(f"{storageloc}importances_{j}{time}{jobid}{modeltype}_classify.txt", "w") as f:
    #         for i in range(len(importances)):
    #             f.write(f"{i}: {importances[i]}\n")
    #
    # test_plot = None
    #
    # if cut_windows:
    #     import matplotlib.pyplot as plt
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(test_labels[0:(LATS*LONS)].reshape(LATS,LONS))
    #     test_plot = test_labels[0:(LATS*LONS)].reshape(LATS,LONS)
    #     plt.show()
    #
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(predictions[0:(LATS*LONS)].reshape(LATS, LONS, order='C'))
    #     plt.show()
    #
    #     plt.subplot(2,3, 1)
    #
    #     plt.imshow(test_labels[0:(LATS*LONS)].reshape(LATS, LONS))
    #     plt.title(f"Test labels for classify")
    #     plt.subplot(2,3, 2)
    #     plt.imshow(predictions[0:(LATS*LONS)].reshape(LATS, LONS, order='C'))
    #     plt.title(f"Predictions for classify")
    #     plt.subplot(2,3, 3)
    #     # plot the errors (preds - labels)
    #     plt.imshow((predictions[0:(LATS*LONS)].reshape(LATS, LONS, order='C') - test_labels[0:(LATS * LONS)].reshape(LATS, LONS)))
    #     plt.title(f"Errors for classify")
    #     plt.subplot(2,3, 4)
    #     # plot the test_data (only the center value of each 11x11 window)
    #     # change the color map for this subplot
    #
    #     if modeltype == "cnn":
    #         plt.imshow(test_data[0:(LATS * LONS),(floor(WINDOWSIZE/2)),(floor(WINDOWSIZE/2))].reshape(LATS, LONS, order='C'), cmap='coolwarm')
    #     elif modeltype == "rf":
    #         plt.imshow(test_data[0:(LATS * LONS), 0].reshape(LATS, LONS, order='C'), cmap='coolwarm')
    #
    #     # title the subplot
    #     plt.title(f"Test data (classify, QT)")
    #
    #     plt.subplot(2,3, 5)
    #     # plot the test_data (only the center value of each 11x11 window)
    #     if modeltype == "cnn":
    #         plt.imshow(test_data_OLD[0:(LATS*LONS),(floor(WINDOWSIZE/2)),(floor(WINDOWSIZE/2))].reshape(LATS, LONS, order='C'), cmap='coolwarm')
    #     elif modeltype == "rf":
    #         plt.imshow(test_data_OLD[0:(LATS*LONS), (floor(WINDOWSIZE/2)),(floor(WINDOWSIZE/2))].reshape(LATS, LONS, order='C'), cmap='coolwarm')
    #
    #     # title the subplot
    #     plt.title(f"Test data (classify, untransformed)")
    #
    #     plt.subplot(2,3, 6)
    #     if modeltype == "cnn":
    #         plt.imshow(train_data[0:(LATS * LONS),(floor(WINDOWSIZE/2)),(floor(WINDOWSIZE/2))].reshape(LATS, LONS, order='C'), cmap='coolwarm')
    #     elif modeltype == "rf":
    #         plt.imshow(train_data[0:(LATS * LONS), 0].reshape(LATS, LONS, order='C'), cmap='coolwarm')
    #
    #     plt.title(f"Train data (classify, untransformed)")
    #
    #     # enlarge the figure so the subplots are not so close to each other
    #     plt.gcf().set_size_inches(20, 10)
    #     # generate a date string for the plot title
    #     # save the figure with that number as a suffix
    #     date = datetime.datetime.now()
    #     date_string = date.strftime("%Y-%m-%d-%H-%M-%S")
    #     if plotdir is not None:
    #         plt.savefig(f"{plotdir}_{j}_{date_string}_{modeltype}{time}{jobid}_classify.png")
    #     plt.clf()
    #
    #
    # if modeltype == "cnn":
    #     with open(f"{storageloc}average_error.txt", "w") as f:
    #         f.write(str(score[1]))
    #
    #     # run_classification(predictions, test_labels)
    #
    #     scores.append(score[1])
    #
    # # save the model
    # try:
    #     if modeltype == "cnn":
    #         model.save(f"{storageloc}model_{metric}{j}{time}{modeltype}{jobid}_classify.h5")
    #         plt.plot(history.history['accuracy'])
    #         plt.plot(history.history['val_accuracy'])
    #         plt.title('model accuracy')
    #         plt.ylabel('accuracy')
    #         plt.xlabel('epoch')
    #         plt.legend(['train', 'test'], loc='upper left')
    #         plt.show()
    #         plt.savefig(f"{storageloc}acc_history_{j}{time}{modeltype}{jobid}_classify.png")
    #         plt.clf()
    # except:
    #     pass
    #
    # with open(f"{storageloc}av_preds_{metric}_{j}{time}{modeltype}{jobid}_classify", "wb") as fp:
    #     pickle.dump(av_preds, fp)
    # with open(f"{storageloc}av_{metric}_{j}{time}{modeltype}{jobid}_classify", "wb") as fp:
    #     pickle.dump(av_dssims, fp)
    # if cut_windows:
    #     predictions = predictions.squeeze().reshape(-1, LATS, LONS, order='C')
    #     # reorder the dimensions of predictions so that the time dimension is last
    #     predictions = np.moveaxis(predictions, 0, -1)
    # # if modeltype == "cnn":
    # np.save(f"{storageloc}predictions_{metric}_{j}{time}{modeltype}{jobid}_classify.npy", label_predictions)
    # np.save(f"{storageloc}test_plot_{j}{time}{modeltype}{jobid}_classify.npy", test_plot)
    # # also save scores, av_preds, av_dssims, predictions, test_plot in .npy files
    # if cut_windows:
    #     labels = test_labels.squeeze().reshape(-1, LATS, LONS, order='C')
    #     # reorder the dimensions of labels so that the time dimension is last
    #     labels = np.moveaxis(labels, 0, -1)
    # if modeltype == "cnn" or modeltype == "rf":
    #     class_labels = label_encoder.inverse_transform(newlabels)
    #     np.save(f"{storageloc}labels_{metric}_{j}{time}{modeltype}{jobid}_classify.npy", class_labels)
    # if modeltype == "cnn":
    #     with open(f"{storageloc}scores_{j}{time}{modeltype}{jobid}_classify", "wb") as fp:  # Pickling
    #         pickle.dump(scores, fp)
    # if modeltype == "cnn":
    #     return scores, model, av_preds, av_dssims, predictions, test_plot
    # elif modeltype == "rf":
    #     if only_data or feature is not None:
    #         return (0,0), None, None, None, None, None
    #     if av_preds == [] or av_dssims == []:
    #         av_preds = [0]
    #         av_dssims = [0]
    #     return (0,0), model, av_preds, av_dssims, predictions, test_plot