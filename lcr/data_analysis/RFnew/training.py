import xarray as xr
import os
from utils import read_parameters_from_json, read_parameters_from_json
from data_processing import cut_spatial_dataset_into_windows, split_data_into_train_val_test
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


LATS = 182
LONS = 288
WINDOWSIZE = 11

def split_data(dataset: xr.Dataset, label: np.ndarray, time: int, nvar: int, testset: str, lats:int, lons:int, cut_windows:bool = True, window_size: int = 11) -> tuple:
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
    num_windows = lats * lons # 182 * 288

    total_data_points = num_windows * time * nvar
    # Calculate the indices corresponding to 60% and 75% of the data
    index_60pct = int(total_data_points * 0.6)
    index_75pct = int(total_data_points * 0.75)
    index_10pct = int(total_data_points * 0.1)
    index_50pct = int(total_data_points * 0.5)

    # use 90% of the data for training, 9% for validation, and 1% for testing
    if testset == "random":
        train_data, test_data, train_labels, test_labels = train_test_split(dataset[0:(num_windows * time * nvar)],
                                                                            label, test_size=0.1)
        val_data, test_data, val_labels, test_labels = train_test_split(test_data, test_labels, test_size=0.1)
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
        test_labels = label[0:(num_windows * time)]

        # This will randomize the training and validation data, an alternative would be to use the last variable(s) for validation
        train_data, val_data, train_labels, val_labels = train_test_split(dataset[(num_windows * time):(num_windows * time * nvar)],
                                                                          label[
                                                                          (num_windows * time):(num_windows * time * nvar)],
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
        train_labels = label[0:index_10pct]

        # Use the calculated number of windows for test and validation
        test_data = dataset[index_10pct:int(index_10pct + num_windows_test)]
        test_labels = label[index_10pct:int(index_10pct + num_windows_test)]
        val_data = dataset[int(index_10pct + num_windows_test):int(index_10pct + num_windows_test + num_windows_val)]
        val_labels = label[int(index_10pct + num_windows_test):int(index_10pct + num_windows_test + num_windows_val)]

        # If there is a remaining time slice, split it between test and validation to preserve the 60-40 split
        if remaining_start_index < total_data_points:
            remaining_windows = total_data_points - remaining_start_index
            split_index = remaining_windows * 90 // 100
            test_data = np.concatenate(
                (test_data, dataset[int(remaining_start_index):int(remaining_start_index + split_index)]))
            test_labels = np.concatenate(
                (test_labels, label[int(remaining_start_index):int(remaining_start_index + split_index)]))
            val_data = np.concatenate((val_data, dataset[int(remaining_start_index + split_index):]))
            val_labels = np.concatenate((val_labels, label[int(remaining_start_index + split_index):]))

        if not cut_windows:
            train_data = dataset[0:int(index_10pct/num_windows)]
            train_labels = label[0:int(index_10pct/num_windows)]
            val_data = dataset[int(index_10pct/num_windows):(int(index_10pct/num_windows)+int(num_windows_val/num_windows))]
            val_labels = label[int(index_10pct/num_windows):(int(index_10pct/num_windows)+int(num_windows_val/num_windows))]
            test_data = dataset[(int(index_10pct/num_windows)+int(num_windows_val/num_windows)):(int(index_10pct/num_windows)+int(num_windows_val/num_windows)+int(num_windows_test/num_windows))]
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
            train_labels = label[0:index_50pct]

            # Use the calculated number of windows for test and validation
            test_data = dataset[index_50pct:int(index_50pct + num_windows_test)]
            test_labels = label[index_50pct:int(index_50pct + num_windows_test)]
            val_data = dataset[int(index_50pct + num_windows_test):int(index_50pct + num_windows_test + num_windows_val)]
            val_labels = label[int(index_50pct + num_windows_test):int(index_50pct + num_windows_test + num_windows_val)]

            # If there is a remaining time slice, split it between test and validation to preserve the 60-40 split
            if remaining_start_index < total_data_points:
                remaining_windows = total_data_points - remaining_start_index
                split_index = remaining_windows * 50 // 100
                test_data = np.concatenate(
                    (test_data, dataset[int(remaining_start_index):int(remaining_start_index + split_index)]))
                test_labels = np.concatenate(
                    (test_labels, label[int(remaining_start_index):int(remaining_start_index + split_index)]))
                val_data = np.concatenate((val_data, dataset[int(remaining_start_index + split_index):]))
                val_labels = np.concatenate((val_labels, label[int(remaining_start_index + split_index):]))
        elif not cut_windows:
            train_data = dataset[0:int(index_50pct/num_windows)]
            train_labels = label[0:int(index_50pct/num_windows)]
            val_data = dataset[int(index_50pct/num_windows):(int(index_50pct/num_windows)+int(num_windows_val/num_windows))]
            val_labels = label[int(index_50pct/num_windows):(int(index_50pct/num_windows)+int(num_windows_val/num_windows))]
            test_data = dataset[(int(index_50pct/num_windows)+int(num_windows_val/num_windows)):(int(index_50pct/num_windows)+int(num_windows_val/num_windows)+int(num_windows_test/num_windows))]
            test_labels = label[(int(index_50pct/num_windows)+int(num_windows_val/num_windows)):(int(index_50pct/num_windows)+int(num_windows_val/num_windows)+int(num_windows_test/num_windows))]


    return train_data, train_labels, val_data, val_labels, test_data, test_labels

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

def train_cnn(dataset: xr.Dataset, labels: np.ndarray, time, varname, nvar, storageloc,
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

    newlabels = classify(f"{j}.json", metric, labels)


    #### SECTIONS TO REVIEW ####

    label_encoder = LabelEncoder()
    integer_encoded_labels = label_encoder.fit_transform(newlabels)

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

    for i in range(time*nvar):
        # dataset[i] = (dataset[i] - np.mean(dataset[i])) / np.std(dataset[i])
        # convert newlabels to a numpy array
        # newlabels = np.array(newlabels)
        newlabels = np.array(integer_encoded_labels)
        train_data, train_labels, val_data, val_labels, test_data, test_labels = split_data(dataset, newlabels, time, nvar, testset, LATS, LONS, cut_windows)

    # check the echosave directory, open trial_results.csv
    # read the column mean_squared_error to find the row with the minimum value
    # then set filter1, filter2, and dropout to the values in that row
    csv_path = os.path.join("echosave/trial_results.csv")
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            lines = f.readlines()
            mse = []
            # remember to skip the first line
            # for line in lines:
            #     mse.append(float(line.split(",")[1]))
            for line in lines[1:]:
                mse.append(float(line.split(",")[8]))
            min_mse = min(mse)
            min_mse_index = mse.index(min_mse) + 1
            filter1 = int(lines[min_mse_index].split(",")[3])
            filter2 = int(lines[min_mse_index].split(",")[4])
            dropout = float(lines[min_mse_index].split(",")[6])
            batch_size = int(lines[min_mse_index].split(",")[5])
            conv_layers = int(lines[min_mse_index].split(",")[7])
    else:
        filter1 = 16
        filter2 = 16
        dropout = 0.25
        batch_size = 32
        conv_layers=2

    if modeltype == "cnn":
        # build the model described above, using the functional API
        i = tf.keras.Input(shape=(np.shape(train_data)[1], np.shape(train_data)[2], 1))
        x = tf.keras.layers.Conv2D(filter1, kernel_size=(2, 2), activation="relu")(i)
        # if conv_layers is 3, add another conv layer
        if conv_layers == 3 or conv_layers == 4:
            x = tf.keras.layers.Conv2D(filter1, kernel_size=(2, 2), activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Conv2D(filter2, kernel_size=(2, 2), activation="relu")(x)
        if conv_layers == 4:
            x = tf.keras.layers.Conv2D(filter2, kernel_size=(2, 2), activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        # this is a classification problem, so the output layer should have len(labels) nodes and use softmax
        outputs = tf.keras.layers.Dense(len(labels), activation="softmax")(x)

        model = tf.keras.Model(inputs=i, outputs=outputs)
        # model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['mean_absolute_error'])
        # compile the model for classification
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

        model.summary()
    elif modeltype == "rf":
        import ldcpy
        model = sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)


    # apply a quantile transformation to each of the lat/lon slices of the data
    # this will make the data more Gaussian
    # first flatten the data
    savelats = train_data.shape[1]
    savelons = train_data.shape[2]
    train_data = train_data.reshape(train_data.shape[0], -1)
    val_data = val_data.reshape(val_data.shape[0], -1)
    test_data_OLD = test_data.reshape(test_data.shape[0], -1)

    if transform == "quantile":
        train_data = quantile_transform(train_data, output_distribution='uniform', copy=True, n_quantiles=10000)
        val_data = quantile_transform(val_data, output_distribution='uniform', copy=True, n_quantiles=10000)
        test_data = quantile_transform(test_data_OLD, output_distribution='uniform', copy=True, n_quantiles=10000)
    # then put the data back into the original shape
    train_data = train_data.reshape(train_data.shape[0], savelats, savelons)
    val_data = val_data.reshape(val_data.shape[0], savelats, savelons)
    test_data = test_data.reshape(test_data.shape[0], savelats, savelons)
    test_data_OLD = test_data_OLD.reshape(test_data_OLD.shape[0], savelats, savelons)


    if modeltype == "rf":
        if feature is not None:
            # train_data_xr = convert_np_to_xr(train_data)
            # dc = ldcpy.Datasetcalcs(train_data_xr.to_array(), train_data_xr.data_type, ["latitude", "longitude"], weighted=False)
            # ns = dc.get_calc("ns_con_var")
            # ew = dc.get_calc("ew_con_var")
            # we_fd = dc.get_calc("w_e_first_differences")
            # ns_fd = dc.get_calc("n_s_first_differences")
            # fftr = dc.get_calc("fftratio")
            # mg = dc.get_calc("magnitude_range")
            # npns = ns.to_numpy()
            # npew = ew.to_numpy()
            # npwe_fd = we_fd.to_numpy()
            # npns_fd = ns_fd.to_numpy()
            # np_fftr = fftr.to_numpy()
            # np_mg = mg.to_numpy()
            # train_data = np.concatenate((npns, npew, npwe_fd, npns_fd, np_fftr, np_mg.reshape(1, np_mg.shape[0])), axis=0)
            # train_data = train_data.transpose()

            for type in ["train", "test"]:
                if type == "test":
                    test_data_xr = convert_np_to_xr(test_data)
                    dc = ldcpy.Datasetcalcs(test_data_xr.to_array(), test_data_xr.data_type, ["latitude", "longitude"], weighted=False)
                elif type == "train":
                    train_data_xr = convert_np_to_xr(train_data)
                    dc = ldcpy.Datasetcalcs(train_data_xr.to_array(), train_data_xr.data_type, ["latitude", "longitude"], weighted=False)
                if feature in ["ns_con_var", "ew_con_var", "w_e_first_differences", "n_s_first_differences", "fftratio", "fftmax", 'w_e_first_differences_max', 'n_s_first_differences_max', 'mean']:
                    ns = dc.get_calc(feature)
                else:
                    dc = ldcpy.Datasetcalcs(train_data_xr.to_array(), train_data_xr.data_type, [], weighted=False)
                    ns = dc.get_single_calc(feature)
                npns = ns.to_numpy()
                # save train_data, train_labels, val_data, val_labels, test_data, test_labels

                np.save(f"{storageloc}{feature}_{metric}_{type}{time}{jobid}_classify.npy", npns)
            exit(0)

        for type in ["train", "test"]:
            list = None
            for f in featurelist:
                if list is None and f != "magnitude_range":
                    list = np.load(f"{storageloc}{f}_{metric}_{type}{time}{jobid}_classify.npy")
                elif list is None and f == "magnitude_range":
                    list = np.load(f"{storageloc}{f}_{metric}_{type}{time}{jobid}_classify.npy")
                    list = list.reshape(1, list.shape[0])
                elif f == "magnitude_range":
                    feat = np.load(f"{storageloc}{f}_{metric}_{type}{time}{jobid}_classify.npy")
                    list = np.concatenate((list, feat.reshape(1, feat.shape[0])), axis=0)
                else:
                    feat = np.load(f"{storageloc}{f}_{metric}_{type}{time}{jobid}_classify.npy")
                    list = np.concatenate((list, feat), axis=0)
                if type == "train":
                    train_data = list
                elif type == "test":
                    test_data = list

        train_data = train_data.transpose()
        test_data = test_data.transpose()


    # save train_data, train_labels, val_data, val_labels, test_data, test_labels
    np.save(f"{storageloc}train_data_CNN11_local_classify.npy", train_data)
    np.save(f"{storageloc}train_labels_{metric}_CNN11_local_classify.npy", train_labels)
    np.save(f"{storageloc}val_data_CNN11_local_classify.npy", val_data)
    np.save(f"{storageloc}val_labels_{metric}_CNN11_local_classify.npy", val_labels)
    np.save(f"{storageloc}test_data_CNN11_local_classify.npy", test_data)
    np.save(f"{storageloc}test_labels_{metric}_CNN11_local_classify.npy", test_labels)

    if only_data:
        exit()

    #plot the first (LATS * LONS) values of test_labels in a 182*288 grid
    # make this the first of a 1x4 subplot
        # fit the model
    if modeltype == "cnn":
        history = model.fit(train_data, train_labels, epochs=1, batch_size=batch_size, validation_data=(val_data, val_labels))
        score = model.evaluate(test_data, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
    elif modeltype == "rf":
        train_data[train_data == -np.inf] = 0
        train_data[train_data == np.inf] = 0
        train_data[train_data == np.nan] = 0
        test_data[test_data == -np.inf] = 0
        test_data[test_data == np.inf] = 0
        train_data[train_data == np.nan] = 0
        val_data[val_data == -np.inf] = 0
        val_data[val_data == np.inf] = 0
        train_data[train_data == np.nan] = 0
        model.fit(train_data, train_labels)

    val_predictions = model.predict(test_data)
    predictions = np.argmax(val_predictions, axis=1)
    label_predictions = label_encoder.inverse_transform(predictions)
    if modeltype == "rf":
        # save the feature importances as {storageloc}importances_{i}_{j.split('.')[0]}{jobid}{model}
        importances = model.feature_importances_
        np.save(f"{storageloc}importances_{j}{time}{jobid}{modeltype}_classify.npy", importances)

    test_plot = None

    if cut_windows:
        import matplotlib.pyplot as plt
        plt.subplot(1, 2, 1)
        plt.imshow(test_labels[0:(LATS*LONS)].reshape(LATS,LONS))
        test_plot = test_labels[0:(LATS*LONS)].reshape(LATS,LONS)
        plt.show()

        plt.subplot(1, 2, 2)
        plt.imshow(predictions[0:(LATS*LONS)].reshape(LATS, LONS, order='C'))
        plt.show()

        plt.subplot(2,3, 1)

        plt.imshow(test_labels[0:(LATS*LONS)].reshape(LATS, LONS))
        plt.title(f"Test labels for classify")
        plt.subplot(2,3, 2)
        plt.imshow(predictions[0:(LATS*LONS)].reshape(LATS, LONS, order='C'))
        plt.title(f"Predictions for classify")
        plt.subplot(2,3, 3)
        # plot the errors (preds - labels)
        plt.imshow((predictions[0:(LATS*LONS)].reshape(LATS, LONS, order='C') - test_labels[0:(LATS * LONS)].reshape(LATS, LONS)))
        plt.title(f"Errors for classify")
        plt.subplot(2,3, 4)
        # plot the test_data (only the center value of each 11x11 window)
        # change the color map for this subplot

        if modeltype == "cnn":
            plt.imshow(test_data[0:(LATS * LONS),(floor(WINDOWSIZE/2)),(floor(WINDOWSIZE/2))].reshape(LATS, LONS, order='C'), cmap='coolwarm')
        elif modeltype == "rf":
            plt.imshow(test_data[0:(LATS * LONS), 0].reshape(LATS, LONS, order='C'), cmap='coolwarm')

        # title the subplot
        plt.title(f"Test data (classify, QT)")

        plt.subplot(2,3, 5)
        # plot the test_data (only the center value of each 11x11 window)
        if modeltype == "cnn":
            plt.imshow(test_data_OLD[0:(LATS*LONS),(floor(WINDOWSIZE/2)),(floor(WINDOWSIZE/2))].reshape(LATS, LONS, order='C'), cmap='coolwarm')
        elif modeltype == "rf":
            plt.imshow(test_data_OLD[0:(LATS*LONS), (floor(WINDOWSIZE/2)),(floor(WINDOWSIZE/2))].reshape(LATS, LONS, order='C'), cmap='coolwarm')

        # title the subplot
        plt.title(f"Test data (classify, untransformed)")

        plt.subplot(2,3, 6)
        if modeltype == "cnn":
            plt.imshow(train_data[0:(LATS * LONS),(floor(WINDOWSIZE/2)),(floor(WINDOWSIZE/2))].reshape(LATS, LONS, order='C'), cmap='coolwarm')
        elif modeltype == "rf":
            plt.imshow(train_data[0:(LATS * LONS), 0].reshape(LATS, LONS, order='C'), cmap='coolwarm')

        plt.title(f"Train data (classify, untransformed)")

        # enlarge the figure so the subplots are not so close to each other
        plt.gcf().set_size_inches(20, 10)
        # generate a date string for the plot title
        # save the figure with that number as a suffix
        date = datetime.datetime.now()
        date_string = date.strftime("%Y-%m-%d-%H-%M-%S")
        if plotdir is not None:
            plt.savefig(f"{plotdir}_{j}_{date_string}_{modeltype}{time}{jobid}_classify.png")
        plt.clf()


    if modeltype == "cnn":
        with open(f"{storageloc}average_error.txt", "w") as f:
            f.write(str(score[1]))

        # run_classification(predictions, test_labels)

        scores.append(score[1])

    # save the model
    try:
        if modeltype == "cnn":
            model.save(f"model_{metric}{j}{time}{modeltype}{jobid}_classify.h5")
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            plt.savefig(f"{storageloc}acc_history_{j}{time}{modeltype}{jobid}_classify.png")
            plt.clf()
    except:
        pass

    with open(f"{storageloc}av_preds_{metric}_{j}{time}{modeltype}{jobid}_classify", "wb") as fp:
        pickle.dump(av_preds, fp)
    with open(f"{storageloc}av_{metric}_{j}{time}{modeltype}{jobid}_classify", "wb") as fp:
        pickle.dump(av_dssims, fp)
    if cut_windows:
        predictions = predictions.squeeze().reshape(-1, LATS, LONS, order='C')
        # reorder the dimensions of predictions so that the time dimension is last
        predictions = np.moveaxis(predictions, 0, -1)
    # if modeltype == "cnn":
    np.save(f"{storageloc}predictions_{metric}_{j}{time}{modeltype}{jobid}_classify.npy", label_predictions)
    np.save(f"{storageloc}test_plot_{j}{time}{modeltype}{jobid}_classify.npy", test_plot)
    # also save scores, av_preds, av_dssims, predictions, test_plot in .npy files
    if cut_windows:
        labels = test_labels.squeeze().reshape(-1, LATS, LONS, order='C')
        # reorder the dimensions of labels so that the time dimension is last
        labels = np.moveaxis(labels, 0, -1)
    if modeltype == "cnn" or modeltype == "rf":
        class_labels = label_encoder.inverse_transform(newlabels)
        np.save(f"{storageloc}labels_{metric}_{j}{time}{modeltype}{jobid}_classify.npy", class_labels)
    if modeltype == "cnn":
        with open(f"{storageloc}scores_{j}{time}{modeltype}{jobid}_classify", "wb") as fp:  # Pickling
            pickle.dump(scores, fp)
    if modeltype == "cnn":
        return scores, model, av_preds, av_dssims, predictions, test_plot
    elif modeltype == "rf":
        if only_data or feature is not None:
            return (0,0), None, None, None, None, None
        if av_preds == [] or av_dssims == []:
            av_preds = [0]
            av_dssims = [0]
        return (0,0), model, av_preds, av_dssims, predictions, test_plot