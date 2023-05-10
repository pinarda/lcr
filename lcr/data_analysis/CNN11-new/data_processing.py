import xarray as xr
import numpy as np
import os
import ldcpy
import sys
import gc
from sklearn.model_selection import train_test_split
os.environ["HDF5_PLUGIN_PATH"]

def cut_spatial_dataset_into_windows(dataset: xr.Dataset, time: int, varname: str, storageloc: str, window_size: int = 11) -> np.ndarray:
    """
    Extracts every possible contiguous lat-lon chunk from the input dataset.

    Parameters:
        dataset (xr.Dataset): The input xarray dataset.
        time (int): The number of time slices to extract from the dataset.
        varname (str): The variable name used to identify the dataset (part of the filename).
        storageloc (str): The location on the filesystem where the data is stored.

    Returns:
        np.ndarray: An array of shape (num_windows, time, window_size, window_size) containing the extracted windows.
    """
    # Check if the chunks have already been saved and load them if they exist.
    chunks_path = f"{storageloc}{varname}_chunks_{time}.npy"
    if os.path.exists(chunks_path):
        pass
        # return np.load(chunks_path)

    # Extract lat and lon values from the dataset.
    lat_vals = dataset.lat.values
    lon_vals = dataset.lon.values

    num_windows = (len(lat_vals) - window_size + 1) * (len(lon_vals) - window_size + 1)

    # Initialize an empty array to store the chunks.
    chunks = np.ndarray(shape=(num_windows, time, window_size, window_size))

    # Convert the dataset to a numpy array for faster slicing.
    d = dataset.to_array().to_numpy()

    # Iterate over lat, lon, and time to extract 11x11 windows.
    idx = 0
    for i in range(len(lat_vals) - (window_size -1) ):
        for j in range(len(lon_vals) - (window_size -1) ):
            for k in range(time):
                chunks[idx, k] = d[:, k, i:i + window_size, j:j + window_size]
            idx += 1

    # Save the chunks as a numpy array.
    np.save(chunks_path, chunks)

    return chunks

def split_data_into_train_val_test(dataset: xr.Dataset, dssim: np.ndarray, time: int, nvar: int, testset: str, comp: int, window_size: int = 11) -> tuple:
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
    num_windows = 50596

    total_data_points = num_windows * time * nvar
    # Calculate the indices corresponding to 60% and 75% of the data
    index_60pct = int(total_data_points * 0.6)
    index_75pct = int(total_data_points * 0.75)

    # use 90% of the data for training, 9% for validation, and 1% for testing
    if testset == "random":
        train_data, test_data, train_labels, test_labels = train_test_split(dataset[0:(num_windows * time * nvar)],
                                                                            dssim[comp], test_size=0.1)
        val_data, test_data, val_labels, test_labels = train_test_split(test_data, test_labels, test_size=0.1)
    elif testset == "oneout":
        # train_data = dataset[0:(50596*time*nvar-1)]
        # train_labels = dssim[comp][0:(50596*time*nvar-1)]
        val_data = dataset[(num_windows * time * nvar - 2):(num_windows * time * nvar - 1)]
        val_labels = dssim[comp][(num_windows * time * nvar - 2):(num_windows * time * nvar - 1)]
        # test_data = dataset[(50596*time*nvar-1):(50596*time*nvar)]
        # test_labels = dssim[comp][(50596*time*nvar-1):(50596*time*nvar)]

        # Currently, this will always make the first time slice of the data the test set for consistency
        test_data = dataset[0:num_windows]
        test_labels = dssim[comp][0:num_windows]
        train_data = dataset[num_windows:(num_windows * time * nvar)]
        train_labels = dssim[comp][num_windows:(num_windows * time * nvar)]
    elif testset == "10pct":
        # use the first 10% of the data for testing
        test_data = dataset[0:int(num_windows * time * nvar * 0.1)]
        test_labels = dssim[comp][0:int(num_windows * time * nvar * 0.1)]
        # Note: this randomizes the training and validation data, an alternative would be to use the last 10% of the data for validation
        train_data, val_data, train_labels, val_labels = train_test_split(
            dataset[int(num_windows * time * nvar * 0.1):(num_windows * time * nvar)],
            dssim[comp][int(num_windows * time * nvar * 0.1):(num_windows * time * nvar)], test_size=0.1)

    elif testset == "1var":
        # leave out a single variable for testing, and use the rest for training and validation
        test_data = dataset[0:(num_windows * time)]
        test_labels = dssim[comp][0:(num_windows * time)]

        # This will randomize the training and validation data, an alternative would be to use the last variable(s) for validation
        train_data, val_data, train_labels, val_labels = train_test_split(dataset[(num_windows * time):(num_windows * time * nvar)],
                                                                          dssim[comp][
                                                                          (num_windows * time):(num_windows * time * nvar)],
                                                                          test_size=0.1)

        # Alternatively, use the last variable(s) for validation
        # val_data = dataset[(50596*time*(nvar-1)):(50596*time*nvar)]
        # val_labels = dssim[comp][(50596*time*(nvar-1)):(50596*time*nvar)]
        # train_data = dataset[(50596*time):(50596*time*(nvar-1))]
        # train_labels = dssim[comp][(50596*time):(50596*time*(nvar-1))]
    elif testset == "60pct":
        # Use the first 60% of the data for training
        train_data = dataset[0:index_60pct]
        train_labels = dssim[comp][0:index_60pct]

        # Use the last 25% of the data for testing and validation
        last_25pct_data = dataset[index_75pct:]
        last_25pct_labels = dssim[comp][index_75pct:]

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
        train_labels = dssim[comp][0:index_60pct]

        # Use the calculated number of windows for test and validation
        test_data = dataset[index_75pct:int(index_75pct + num_windows_test)]
        test_labels = dssim[comp][index_75pct:int(index_75pct + num_windows_test)]
        val_data = dataset[int(index_75pct + num_windows_test):int(index_75pct + num_windows_test + num_windows_val)]
        val_labels = dssim[comp][int(index_75pct + num_windows_test):int(index_75pct + num_windows_test + num_windows_val)]

        # If there is a remaining time slice, split it between test and validation to preserve the 60-40 split
        if remaining_start_index < total_data_points:
            remaining_windows = total_data_points - remaining_start_index
            split_index = remaining_windows * 40 // 100
            test_data = np.concatenate(
                (test_data, dataset[int(remaining_start_index):int(remaining_start_index + split_index)]))
            test_labels = np.concatenate(
                (test_labels, dssim[comp][int(remaining_start_index):int(remaining_start_index + split_index)]))
            val_data = np.concatenate((val_data, dataset[int(remaining_start_index + split_index):]))
            val_labels = np.concatenate((val_labels, dssim[comp][int(remaining_start_index + split_index):]))


    return train_data, train_labels, val_data, val_labels, test_data, test_labels

def get_object_size_in_bytes(input_obj) -> int:
    """
    Calculates the total memory size (in bytes) of a given Python object, including the memory size
    of all objects referenced by the input object.

    Parameters:
        input_obj: The input Python object whose memory size is to be calculated.

    Returns:
        int: The total memory size (in bytes) of the input object and all objects referenced by it.
    """
    memory_size = 0
    ids = set()
    objects = [input_obj]
    while objects:
        new = []
        for obj in objects:
            if id(obj) not in ids:
                ids.add(id(obj))
                memory_size += sys.getsizeof(obj)
                new.append(obj)
        objects = gc.get_referents(*new)
    return memory_size