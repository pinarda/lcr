import os
import xarray as xr
import numpy as np
import sys
import tensorflow as tf
import pickle
from math import floor
import matplotlib.pyplot as plt
tf.keras.backend.clear_session()
import cftime
import time as ttime

from utils import parse_command_line_arguments, read_parameters_from_json
from data_processing import cut_spatial_dataset_into_windows, split_data_into_train_val_test
from sklearn.preprocessing import quantile_transform
import sklearn
import datetime
import pandas as pd
# import layers
# import random forest regressor
from sklearn.ensemble import RandomForestRegressor
os.environ["HDF5_PLUGIN_PATH"]
from classification_labels import classify
from training import train_cnn

LATS = 182
LONS = 288
WINDOWSIZE = 11

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


def train_cnn_for_dssim_regression(dataset: xr.Dataset, dssim: np.ndarray, time, varname, nvar, storageloc,
                                   testset="random", j=None, plotdir=None, window_size=WINDOWSIZE, only_data=False, modeltype="cnn", feature=None, featurelist=None, transform="quantile", jobid=0, cut_windows=True, metric="dssim") -> float:
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


    scores = []
    av_preds = []
    av_dssims = []
    for comp in dssim.keys():
        model_path = f"{storageloc}model_{metric}{j}{comp}{time}{modeltype}{jobid}.h5"
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            with open(f"{storageloc}av_preds_{metric}_{j}{comp}{time}{modeltype}{jobid}", "rb") as f:
                av_preds = pickle.load(f)
            with open(f"{storageloc}av_{metric}_{j}{comp}{time}{modeltype}{jobid}", "rb") as f:
                av_dssims = pickle.load(f)
            with open(f"{storageloc}predictions_{metric}_{j}{comp}{time}{modeltype}{jobid}.npy", "rb") as f:
                predictions = np.load(f)
            with open(f"{storageloc}test_plot_{j}{comp}{time}{modeltype}{jobid}.npy", "rb") as f:
                test_plot = np.load(f, allow_pickle=True)
            if modeltype == "cnn":
                with open(f"{storageloc}scores_{j}{comp}{time}{modeltype}{jobid}", "rb") as f:
                    scores = pickle.load(f)
        else:
            # perform a quanitle transformation to make the data more Gaussian using the sklearn library
            for i in range(time*nvar):
                # dataset[i] = (dataset[i] - np.mean(dataset[i])) / np.std(dataset[i])
                train_data, train_labels, val_data, val_labels, test_data, test_labels = split_data_into_train_val_test(dataset, dssim, time, nvar, testset, comp, LATS, LONS, cut_windows)

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
                outputs = tf.keras.layers.Dense(1, activation="linear")(x)
                model = tf.keras.Model(inputs=i, outputs=outputs)
                model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['mean_absolute_error'])
                model.summary()
            elif modeltype == "rf":
                import ldcpy
                model = sklearn.ensemble.RandomForestRegressor(n_estimators=10, max_depth=10)


            # apply a quantile transformation to each of the lat/lon slices of the data
            # this will make the data more Gaussian
            # first flatten the data
            savelats = train_data.shape[1]
            savelons = train_data.shape[2]
            train_data = train_data.reshape(train_data.shape[0], -1)
            # val_data = val_data.reshape(val_data.shape[0], -1)
            test_data_OLD = test_data.reshape(test_data.shape[0], -1)

            if transform == "quantile":
                train_data = quantile_transform(train_data, output_distribution='uniform', copy=True, n_quantiles=10000)
                # val_data = quantile_transform(val_data, output_distribution='uniform', copy=True, n_quantiles=10000)
                test_data = quantile_transform(test_data_OLD, output_distribution='uniform', copy=True, n_quantiles=10000)
            # then put the data back into the original shape
            train_data = train_data.reshape(train_data.shape[0], savelats, savelons)
            # val_data = val_data.reshape(val_data.shape[0], WINDOWSIZE, WINDOWSIZE)
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

                        np.save(f"{storageloc}{feature}_{metric}_{type}{time}{comp}{jobid}.npy", npns)
                    continue

                for type in ["train", "test"]:
                    list = None
                    for f in featurelist:
                        if list is None and f != "magnitude_range":
                            list = np.load(f"{storageloc}{f}_{metric}_{type}{time}{comp}{jobid}.npy")
                        elif list is None and f == "magnitude_range":
                            list = np.load(f"{storageloc}{f}_{metric}_{type}{time}{comp}{jobid}.npy")
                            list = list.reshape(1, list.shape[0])
                        elif f == "magnitude_range":
                            feat = np.load(f"{storageloc}{f}_{metric}_{type}{time}{comp}{jobid}.npy")
                            list = np.concatenate((list, feat.reshape(1, feat.shape[0])), axis=0)
                        else:
                            feat = np.load(f"{storageloc}{f}_{metric}_{type}{time}{comp}{jobid}.npy")
                            list = np.concatenate((list, feat), axis=0)
                        if type == "train":
                            train_data = list
                        elif type == "test":
                            test_data = list

                train_data = train_data.transpose()
                test_data = test_data.transpose()


            # save train_data, train_labels, val_data, val_labels, test_data, test_labels
            np.save(f"{storageloc}train_data_CNN11_local.npy", train_data)
            np.save(f"{storageloc}train_labels_{metric}_CNN11_local.npy", train_labels)
            np.save(f"{storageloc}val_data_CNN11_local.npy", val_data)
            np.save(f"{storageloc}val_labels_{metric}_CNN11_local.npy", val_labels)
            np.save(f"{storageloc}test_data_CNN11_local.npy", test_data)
            np.save(f"{storageloc}test_labels_{metric}_CNN11_local.npy", test_labels)

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

            predictions = model.predict(test_data)
            average_prediction = np.mean(predictions)
            average_dssim = np.mean(test_labels)
            if modeltype == "rf":
                # save the feature importances as {storageloc}importances_{i}_{j.split('.')[0]}{jobid}{model}
                importances = model.feature_importances_
                np.save(f"{storageloc}importances_{j}{comp}{time}{jobid}{modeltype}.npy", importances)

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

                print("Average prediction: ", average_prediction)
                print("Average dssim: ", average_dssim)

                plt.subplot(2,3, 1)

                plt.imshow(test_labels[0:(LATS*LONS)].reshape(LATS, LONS))
                plt.title(f"Test labels for {comp}")
                plt.subplot(2,3, 2)
                plt.imshow(predictions[0:(LATS*LONS)].reshape(LATS, LONS, order='C'))
                plt.title(f"Predictions for {comp}")
                plt.subplot(2,3, 3)
                # plot the errors (preds - labels)
                plt.imshow((predictions[0:(LATS*LONS)].reshape(LATS, LONS, order='C') - test_labels[0:(LATS * LONS)].reshape(LATS, LONS)))
                plt.title(f"Errors for {comp}")
                plt.subplot(2,3, 4)
                # plot the test_data (only the center value of each 11x11 window)
                # change the color map for this subplot

                if modeltype == "cnn":
                    plt.imshow(test_data[0:(LATS * LONS),(floor(WINDOWSIZE/2)),(floor(WINDOWSIZE/2))].reshape(LATS, LONS, order='C'), cmap='coolwarm')
                elif modeltype == "rf":
                    plt.imshow(test_data[0:(LATS * LONS), 0].reshape(LATS, LONS, order='C'), cmap='coolwarm')

                # title the subplot
                plt.title(f"Test data ({comp}, QT)")

                plt.subplot(2,3, 5)
                # plot the test_data (only the center value of each 11x11 window)
                if modeltype == "cnn":
                    plt.imshow(test_data_OLD[0:(LATS*LONS),(floor(WINDOWSIZE/2)),(floor(WINDOWSIZE/2))].reshape(LATS, LONS, order='C'), cmap='coolwarm')
                elif modeltype == "rf":
                    plt.imshow(test_data_OLD[0:(LATS*LONS), (floor(WINDOWSIZE/2)),(floor(WINDOWSIZE/2))].reshape(LATS, LONS, order='C'), cmap='coolwarm')

                # title the subplot
                plt.title(f"Test data ({comp}, untransformed)")

                plt.subplot(2,3, 6)
                if modeltype == "cnn":
                    plt.imshow(train_data[0:(LATS * LONS),(floor(WINDOWSIZE/2)),(floor(WINDOWSIZE/2))].reshape(LATS, LONS, order='C'), cmap='coolwarm')
                elif modeltype == "rf":
                    plt.imshow(train_data[0:(LATS * LONS), 0].reshape(LATS, LONS, order='C'), cmap='coolwarm')

                plt.title(f"Train data ({comp}, untransformed)")

                # enlarge the figure so the subplots are not so close to each other
                plt.gcf().set_size_inches(20, 10)
                # generate a date string for the plot title
                # save the figure with that number as a suffix
                date = datetime.datetime.now()
                date_string = date.strftime("%Y-%m-%d-%H-%M-%S")
                if plotdir is not None:
                    plt.savefig(f"{plotdir}{comp}_{j}_{date_string}_{modeltype}{time}{jobid}.png")
                plt.clf()


            if modeltype == "cnn":
                with open(f"{storageloc}average_error.txt", "w") as f:
                    f.write(str(score[1]))

                # run_classification(predictions, test_labels)

                scores.append(score[1])
            av_preds.append(average_prediction)
            av_dssims.append(average_dssim)

            # save the model
            try:
                if modeltype == "cnn":
                    model.save(f"{storageloc}model_{metric}{j}{comp}{time}{modeltype}{jobid}.h5")
                    plt.plot(history.history['accuracy'])
                    plt.plot(history.history['val_accuracy'])
                    plt.title('model accuracy')
                    plt.ylabel('accuracy')
                    plt.xlabel('epoch')
                    plt.legend(['train', 'test'], loc='upper left')
                    plt.show()
                    plt.savefig(f"{storageloc}acc_history_{j}{comp}{time}{modeltype}{jobid}.png")
                    plt.clf()
            except:
                pass

            with open(f"{storageloc}av_preds_{metric}_{j}{comp}{time}{modeltype}{jobid}", "wb") as fp:
                pickle.dump(av_preds, fp)
            with open(f"{storageloc}av_{metric}_{j}{comp}{time}{modeltype}{jobid}", "wb") as fp:
                pickle.dump(av_dssims, fp)
            if cut_windows:
                predictions = predictions.squeeze().reshape(-1, LATS, LONS, order='C')
                # reorder the dimensions of predictions so that the time dimension is last
                predictions = np.moveaxis(predictions, 0, -1)
            # if modeltype == "cnn":
            np.save(f"{storageloc}predictions_{metric}_{j}{comp}{time}{modeltype}{jobid}.npy", predictions)
            np.save(f"{storageloc}test_plot_{j}{comp}{time}{modeltype}{jobid}.npy", test_plot)
            # also save scores, av_preds, av_dssims, predictions, test_plot in .npy files
            if cut_windows:
                labels = test_labels.squeeze().reshape(-1, LATS, LONS, order='C')
                # reorder the dimensions of labels so that the time dimension is last
                labels = np.moveaxis(labels, 0, -1)
            if modeltype == "cnn" or modeltype == "rf":
                if cut_windows:
                    np.save(f"{storageloc}labels_{metric}_{j}{comp}{time}{modeltype}{jobid}.npy", labels)
            if modeltype == "cnn":
                with open(f"{storageloc}scores_{j}{comp}{time}{modeltype}{jobid}", "wb") as fp:  # Pickling
                    pickle.dump(scores, fp)
    if modeltype == "cnn":
        return scores, model, av_preds, av_dssims, predictions, test_plot
    elif modeltype == "rf":
        if only_data or feature is not None:
            return (0,0), None, None, None, None, None
        return (0,0), model, av_preds, av_dssims, predictions, test_plot

def create_classification_matrix(predicted_dssims, true_dssims, threshold = 0.9995):
    """
    Create a classification matrix based on predicted and true DSSIM values.

    Parameters:
        predicted_dssims (np.ndarray): The predicted DSSIM values.
        true_dssims (np.ndarray): The true DSSIM values.

    Returns:
        np.ndarray: The classification matrix.
    """
    # Calculate the average DSSIM for each time slice using the true values
    true_averages = np.mean(true_dssims, axis=1)

    # Calculate the average DSSIM for each time slice using the predicted values
    predicted_averages = np.mean(predicted_dssims, axis=1)

    # Compare true_averages and predicted_averages to the threshold
    true_over_threshold = true_averages > threshold
    predicted_over_threshold = predicted_averages > threshold

    # Calculate the classification matrix
    TP = np.sum(np.logical_and(true_over_threshold, predicted_over_threshold))
    TN = np.sum(np.logical_and(~true_over_threshold, ~predicted_over_threshold))
    FP = np.sum(np.logical_and(~true_over_threshold, predicted_over_threshold))
    FN = np.sum(np.logical_and(true_over_threshold, ~predicted_over_threshold))

    classification_matrix = np.array([[TP, FP], [FN, TN]])

    print("Classification Matrix:")
    print(classification_matrix)

    return classification_matrix

def create_cftime_range(start_year, periods, calendar='noleap'):
    return [cftime.DatetimeNoLeap(start_year + i, 1, 1) for i in range(periods)]

def build_model_and_evaluate_performance(timeoverride=None, j=0, name="", stride=1,
                                         only_data=False, modeltype="cnn", metric="dssim",
                                         featureoverride=True, feature=None, json=None, testset="random",
                                         featurelist=None, xform="quantile", jobid=0, labelsonly=False):
    # args = parse_command_line_arguments()
    #
    # json = args.json
    # testset = args.testset
    # f = args.feature
    # featurelist = args.listfeatures
    # xform = args.transform
    # jobid = args.jobid
    # This version of the main function builds a single CNN on all variables, useful for training to predict a new variable
    # read in the scratch.json configuration file that specifies the location of the datasets
    save, vlist, pres, posts, opath, cpath, cdirs, ldcpypath, time, storageloc, navg, stride, m, cut_dataset, subdirs = read_parameters_from_json(json)
    metric = metric

    if featureoverride:
        feature=feature
    else:
        feature=feature



    if timeoverride is not None:
        time = timeoverride
    if ldcpypath:
        sys.path.insert(0, ldcpypath)
    import ldcpy
    # create a list of compressed paths by prepending cpath to each directory in cdirs
    newcdirs = []
    newopaths = []
    for cdir in cdirs:
        for dir in subdirs:
            newcdirs.append(dir + "/" + cdir)

    # Split the string
    parts = opath.rsplit('/', 2)

    # First part of the split (path without the last directory)
    first_part = parts[0] + '/'

    # Second part of the split (last directory)
    second_part = '/' + parts[1] + '/'
    for dir in subdirs:
        newopaths.append(first_part + dir + second_part)


    cpaths = [cpath + cdir for cdir in newcdirs]
    # add opath to the beginning of cpaths

    paths = newopaths + cpaths

    # this is not a good way to initialize final_cut_dataset_orig
    final_cut_dataset_orig = np.empty((0, WINDOWSIZE, WINDOWSIZE))
    final_cut_dataset_zfp = np.empty((0, WINDOWSIZE, WINDOWSIZE))
    final_dssim_mats = {}
    average_dssims = {}
    for varname in vlist:
        files = []
        for path in paths:
            # create a list of files to open by appending pre, varname, and post to the filename
            # for every variable in vlist
            print(len(subdirs))
            print(pres)
            for i in range(len(subdirs)):
                if path.rsplit('/', 3)[1] == subdirs[i] or path.rsplit('/', 3)[2] == subdirs[i]:
                    if path.rsplit('/', 3)[2] == "orig":
                        files.append(f"{path}" + pres[i] + varname + posts[i])
                    else:
                        files.append(f"{path}/" + pres[i] + varname + posts[i])

            # create a list of labels by adding "orig" to the list of cdirs
        labels = ["orig" + dir for dir in subdirs] + newcdirs
        # print(files)
        dataset_col = ldcpy.open_datasets(list_of_files=files,
                                          labels=labels,
                                          data_type="cam-fv",
                                          varnames=[varname],
                                          chunks={"time": 50})
        # print the length of the time dimension
        # print(len(dataset_col.time))

        # Determine the maximum length of continuous data across all collections
        # Determine the maximum length of continuous data across all collections
        max_len = max(dataset_col.isel(collection=i).dropna(dim='time', how='all').dims['time'] for i in
                      range(len(dataset_col.collection)))

        # Create a new cftime range for the continuous time dimension
        new_time_dim = create_cftime_range(1920, max_len, calendar='noleap')  # Adjust as needed

        # Prepare an empty DataArray or Dataset with the correct dimensions, including lat and lon
        realigned_data = xr.Dataset({
            varname: (('collection', 'time', 'lat', 'lon'), np.full(
                (len(dataset_col.collection), max_len, len(dataset_col.lat), len(dataset_col.lon)), np.nan)),
        }, coords={
            'collection': dataset_col.collection,
            'time': new_time_dim,
            'lat': dataset_col.lat,
            'lon': dataset_col.lon,
        })

        # Copy and realign the data for each collection element
        for i in range(len(dataset_col.collection)):
            element_data = dataset_col.isel(collection=i).dropna(dim='time', how='all')

            # Ensure lat and lon coordinates are included without modification
            # The realignment focuses on the 'time' dimension; 'lat' and 'lon' remain as in the original dataset
            for t_idx, t_val in enumerate(element_data.time):
                realigned_data[varname][i, t_idx, :, :] = element_data[varname].sel(time=t_val).values

        timeloc = int(time * 0.2)
        # realigned_data = realigned_data.transpose('lat', 'lon', "time", "collection")

        if not labelsonly:
            ldcpy.plot(dataset_col, varname, "mean", labels, start=timeloc, end=timeloc, weighted=False)
            plt.savefig(f"{storageloc}_data_{time}_{name}_{modeltype}_{varname}.png", bbox_inches='tight')

            ldcpy.plot(dataset_col, varname, "mean", labels, start=timeloc, end=timeloc, calc_type="diff", weighted=False)
            plt.savefig(f"{storageloc}_data_diff_{time}_{name}_{modeltype}_{varname}.png", bbox_inches='tight')

        # extract the original and compressed dataarrays

        dataset_orig = realigned_data.sel(collection=["orig" + dir for dir in subdirs]).to_array().squeeze()
        # pad the longitude dimension of the original dataset by 5 on each side (wrap around)
        dataset_orig = xr.concat([dataset_orig[:, :, :, (-1 * floor(WINDOWSIZE/2)):], dataset_orig, dataset_orig[:, :, :, :(floor(WINDOWSIZE/2))]], dim="lon")
        dssim_mats = {}
        # roll orig dataset using the xarray roll function
        # num = 0
        # dataset_orig = dataset_orig.roll(lat=num, roll_coords=True)
        for m in metric:
            dssim_mats[m] = {}
            for cdir in cdirs:
                average_dssims[cdir] = {}

                dataset_zfp = realigned_data.sel(collection=[dir + "/" + cdir for dir in subdirs]).to_array().squeeze()
                # pad the longitude dimension of the compressed dataset by 5 on each side (wrap around)
                dataset_zfp = xr.concat([dataset_zfp[:, :, :, (-1 * floor(WINDOWSIZE/2)):], dataset_zfp, dataset_zfp[:, :, :, :(floor(WINDOWSIZE/2))]], dim="lon")
                # dataset_zfp = dataset_zfp.roll(lat=num, roll_coords=True)
                # dssim_mats[cdir] = np.empty((len(subdirs), time, (LATS * (LONS+2*(WINDOWSIZE-11)))))
                dssim_mats[m][cdir] = {}

                for dir in subdirs:
                    average_dssims[cdir][dir] = np.empty((time))
                    dssim_mats[m][cdir][dir] = np.empty((time, (LATS * (LONS+2*(WINDOWSIZE-11)))))

                    for t in range(0, time):
                        # print the length of the time dimension
                        print(len(dataset_orig.time))
                        print(len(dataset_zfp.time))
                        dc = ldcpy.Diffcalcs(dataset_orig.sel(collection=("orig" + dir)).isel(time=t*stride), dataset_zfp.sel(collection=(dir + "/" + cdir)).isel(time=t*stride), data_type="cam-fv")

                        if "dssim" in metric:
                            average_dssims[cdir][dir][t] = dc.get_diff_calc("ssim_fp", xsize=11, ysize=11)
                            dssim_mats[m][cdir][dir][t] = dc._ssim_mat_fp[0].flatten()
                        elif "mse" in metric:
                            dc2 = ldcpy.Datasetcalcs(dataset_orig.isel(time=t*stride) - dataset_zfp.isel(time=t*stride), data_type="cam-fv", aggregate_dims=[])
                            mse = dc2.get_calc("mean_squared")
                            # ignore the top and bottom 5 rows and columns
                            mse = mse[(floor(WINDOWSIZE/2)):(-1 * floor(WINDOWSIZE/2)), (floor(WINDOWSIZE/2)):(-1 * floor(WINDOWSIZE/2))]
                            dssim_mats[m][cdir][dir][t] = mse.to_numpy().flatten()
                        elif "logdssim" in metric:
                            dc.get_diff_calc("ssim_fp")
                            dc._ssim_mat_fp[0] = np.log(1 - dc._ssim_mat_fp[0])
                            dssim_mats[m][cdir][dir][t] = dc._ssim_mat_fp[0].flatten()



        # cut_dataset_orig = cut_spatial_dataset_into_windows(dataset_col.sel(collection="orig").roll(lat=num, roll_coords=True), time, varname, storageloc)
        if cut_dataset:
            cut_dataset_orig = cut_spatial_dataset_into_windows(dataset_col.sel(collection=["orig" + dir for dir in subdirs]), time, varname, storageloc, window_size=WINDOWSIZE, nsubdirs=len(subdirs))
            cut_dataset_zfp = cut_spatial_dataset_into_windows(dataset_col.sel(collection=[dir + "/" + cdir for dir in subdirs]), time, varname, storageloc, window_size=WINDOWSIZE, nsubdirs=len(subdirs))

            # -1 means unspecified (should normally be (LATS * LONS) * time unless
            # the number of time steps loaded by cut_dataset is different than time)

            cut_dataset_orig = cut_dataset_orig.reshape((-1, WINDOWSIZE, WINDOWSIZE), order="F")
            cut_dataset_zfp = cut_dataset_zfp.reshape((-1, WINDOWSIZE, WINDOWSIZE), order="F")
            # flatten dssim_mats over time
            for m in metric:
                for i, cdir in enumerate(cdirs):
                    for j, dir in enumerate(subdirs):
                        print(m)
                        print (cdir)
                        print (dir)
                        print (type(dssim_mats))
                        print (type(dssim_mats[m]))
                        print (type(dssim_mats[m][cdir]))
                        print (type(dssim_mats[m][cdir][dir]))

                        print (dssim_mats[m][cdir][dir].shape)
                        dssim_mats[m][cdir][dir] = dssim_mats[m][cdir][dir].flatten()
                    # dssim_mats[cdir][dir] = dssim_mats[cdir][dir].flatten()
                # stack the dssim_mats for each compression level
                    dssim_mats[m][cdir] = np.stack([dssim_mats[m][cdir][dir] for dir in subdirs], axis=0).flatten(order="C")


                np.save(f"{storageloc}{varname}_{metric}_mat_{time}_{j}.npy", dssim_mats)
                # if not os.path.exists(f"{storageloc}{varname}_chunks_{time}.npy"):
                #     np.save(f"{storageloc}{varname}_chunks_{time}.npy", cut_dataset_orig)

                    # hopefully, by this point dssim_mat contains all the dssims for a single variable at every compression level
                    # and cut_dataset_orig contains all the uncompressed 11x11 chunks for a single variable
                #append cut_dataset_orig to final_cut_dataset_orig
                final_cut_dataset_orig = np.append(final_cut_dataset_orig, cut_dataset_orig[0:((LATS * LONS)*time*len(subdirs))], axis=0)
                final_cut_dataset_zfp = np.append(final_cut_dataset_zfp, cut_dataset_zfp[0:((LATS * LONS)*time*len(subdirs))], axis=0)

                # turn final_cut_dataset_orig and final_cut_dataset_zfp into xarrays
                final_cut_dataset_orig_xr = convert_np_to_xr(final_cut_dataset_orig).array
                final_cut_dataset_zfp_xr = convert_np_to_xr(final_cut_dataset_zfp).array

            if "dssim" not in metric:
                dssim_mats = {}
                for m in metric:
                    dssim_mats[m] = {}

            for cdir in cdirs:
                # for t in range(final_cut_dataset_orig_xr.sizes["time"]):
                dc2 = ldcpy.Diffcalcs(final_cut_dataset_orig_xr,
                                      final_cut_dataset_zfp_xr, data_type="cam-fv", aggregate_dims=["latitude", "longitude"])
                if "pcc" in metric:
                    mat_xr = dc2.get_diff_calc("pearson_correlation_coefficient")
                    dssim_mats["pcc"][cdir] = mat_xr.to_numpy()
                elif "ks" in metric:
                    mat_xr = dc2.get_diff_calc("ks_p_value")
                    dssim_mats["ks"][cdir] = mat_xr.to_numpy()
                elif "spre" in metric:
                    mat_xr = dc2.get_diff_calc("spatial_rel_error")
                    dssim_mats["spre"][cdir] = mat_xr.to_numpy()



            # compute the average dssims by averaging each consecutive 182*288 (or whatever) dssim_mats
            # for t in range(time)
            # average_dssims = np.mean(dssim_mats[LATS*LONS*t??], axis=1)

            #append dssim_mats to final_dssim_mats
            for m in metric:
                final_dssim_mats[m] = {}

            for cdir in cdirs:
                for m in metric:
                    if cdir not in final_dssim_mats:
                        final_dssim_mats[m][cdir] = dssim_mats[m][cdir]
                    else:
                        final_dssim_mats[m][cdir] = np.append(final_dssim_mats[m][cdir], dssim_mats[m][cdir], axis=0)

        #append all elements in average_dssims to a single array
        # average_dssims = np.array([average_dssims[cdir] for cdir in cdirs])

        # call fit_cnn on the 11x11 chunks and the dssim values
        fname = json.split(".")[0]

        # start timer
        start = ttime.time()


    if labelsonly:
        dar = np.array(dataset_orig)[0:(time * len(subdirs))]
        dark = dar.reshape(-1, *dar.shape[2:])
        new_dict = {}
        for key, sub_dict in average_dssims.items():
            # Flatten all arrays in the sub-dictionary and concatenate them
            flattened_array = np.concatenate([v for v in sub_dict.values()])
            # Add to the new dictionary under the original key
            new_dict[key] = flattened_array
        if feature is not None:
            train_cnn(dark, new_dict, time * len(subdirs), "combine", len(vlist),
                                           storageloc, testset, fname, only_data=only_data, modeltype=modeltype,
                                           plotdir=save,
                                           feature=feature, featurelist=featurelist, cut_windows=False, jobid=jobid,
                                           metric=metric)
            return
        errors, model, av_preds, av_dssims, predictions, test_dssims = train_cnn(dark, new_dict,
                                                                                  time * len(
                                                                                      subdirs),
                                                                                  "combine",
                                                                                  len(vlist),
                                                                                  storageloc,
                                                                                  testset, fname,
                                                                                  only_data=only_data,
                                                                                  modeltype=modeltype,
                                                                                  plotdir=save,
                                                                                  feature=feature,
                                                                                  featurelist=featurelist,
                                                                                  transform=xform,
                                                                                  jobid=jobid,
                                                                                  cut_windows=False,
                                                                                  metric=metric)

    else:
        if cut_dataset:
            if feature is not None:
                train_cnn_for_dssim_regression(final_cut_dataset_orig, final_dssim_mats, time*len(subdirs), "combine", len(vlist),
                                               storageloc, testset, fname, only_data=only_data, modeltype=modeltype, plotdir=save,
                                               feature=feature, featurelist=featurelist, jobid=jobid, metric=metric, cut_windows=cut_dataset)
                return
            errors, model, av_preds, av_dssims, predictions, test_dssims = train_cnn_for_dssim_regression(final_cut_dataset_orig, final_dssim_mats, time*len(subdirs), "combine", len(vlist), storageloc, testset, fname, only_data=only_data, modeltype=modeltype, plotdir=save, feature=feature, featurelist=featurelist, transform=xform, jobid=jobid, metric=metric, cut_windows=cut_dataset)
        else:
            dar = np.array(dataset_orig)[0:(time * len(subdirs))]
            dark = dar.reshape(dar.shape[0] * dar.shape[1], LATS + 10, LONS + 10, order="F")
            new_dict = {}
            for key, sub_dict in average_dssims.items():
                # Flatten all arrays in the sub-dictionary and concatenate them
                flattened_array = np.concatenate([v for v in sub_dict.values()])
                # Add to the new dictionary under the original key
                new_dict[key] = flattened_array
            if feature is not None:
                train_cnn_for_dssim_regression(dark, new_dict, time*len(subdirs), "combine", len(vlist),
                                               storageloc, testset, fname, only_data=only_data, modeltype=modeltype, plotdir=save,
                                               feature=feature, featurelist=featurelist, cut_windows=False, jobid=jobid, metric=metric)
                return
            errors, model, av_preds, av_dssims, predictions, test_dssims = train_cnn_for_dssim_regression(dark, new_dict, time*len(subdirs), "combine", len(vlist), storageloc, testset, fname, only_data=only_data, modeltype=modeltype, plotdir=save, feature=feature, featurelist=featurelist, transform=xform, jobid=jobid, cut_windows=False, metric=metric)

    print(errors)
    # grab the first (LATS * LONS) dssims for each compression level from dssim_mats
    # dssim_mats = {cdir: dssim_mats[cdir][0:(LATS * LONS)] for cdir in cdirs}
    preds_files = {}
    dssims_files = {}

    for m in metric:
        dssims_files[m] = {}
        for cdir in cdirs:
            if cut_dataset:
                final = final_dssim_mats[m][cdir][0:(LATS * LONS)].reshape((LATS, LONS))
            else:
                final = new_dict
            if type(time) is list:
                for t in time:
                    np.save(f"{storageloc}{cdir}_{metric}_mat_{t}_{name}{jobid}.npy", final)
                    # also save the predictions
                    # and the errors
                    preds = np.zeros((LATS, LONS)).flatten()
                    # set the values of mymap to the first (LATS * LONS) values of predictions
                    if len(predictions) < (LATS * LONS):
                        preds[0:(len(predictions))] = predictions.squeeze()
                        preds = preds.reshape((LATS, LONS))
                    else:
                        preds = predictions.squeeze()[0:(LATS * LONS)].reshape((LATS, LONS))
                    np.save(f"{storageloc}{cdir}_{metric}_preds_{t*len(subdirs)}_{name}{jobid}.npy", preds)
            else:
                np.save(f"{storageloc}{cdir}_{metric}_mat_{time*len(subdirs)}_{name}{jobid}.npy", final)
                if cut_dataset:
                    np.save(f"{storageloc}{cdir}_{metric}_mat_alltime_{time*len(subdirs)}_{name}{jobid}.npy", final_dssim_mats[m][cdir].reshape(LATS, (LONS+2*(WINDOWSIZE-11)), -1))
                # also save the predictions
                # and the errors
                # preds = np.zeros((LATS, LONS)).flatten()
                # set the values of mymap to the first (LATS * LONS) values of predictions
                # if len(predictions) < (LATS * LONS):
                #     preds[0:(len(predictions))] = predictions.squeeze()
                # else:
                    # preds = predictions.squeeze()[0:(LATS * LONS)].reshape((LATS, LONS))
                # np.save(f"{storageloc}{cdir}_preds_{time}_{name}.npy", preds)
                # np.save(f"{storageloc}{cdir}_preds_mat_alltime_{name}.npy", predictions.squeeze().reshape(LATS, LONS, -1))
                # preds_file = f"{storageloc}{cdir}_preds_mat_alltime_{name}.npy"
                dssims_file = f"{storageloc}{cdir}_{metric}_mat_alltime_{time*len(subdirs)}_{name}{jobid}.npy"
                # preds_files[cdir] = preds_file
                dssims_files[m][cdir] = dssims_file


    return errors, av_preds, av_dssims, test_dssims, dssims_files

def build_and_evaluate_models_for_time_slices(times, j, name, only_data=False, modeltype="cnn",
                                              feature=None, metric="dssim", json=None, testset="random",
                                                featurelist=None, xform="quantile", jobid=0, labelsonly=False):
    # Will need to perform a normalization step.

    errors = []
    av_preds = []
    av_dssims = []
    dssim_fs = {}

    for i in times:
        if only_data or feature:
            build_model_and_evaluate_performance(i, j, name, only_data=only_data, modeltype=modeltype, metric=metric, feature=feature, json=json, testset=testset, featurelist=featurelist, xform=xform, jobid=jobid, labelsonly=True)
            continue
        e, p, d, test_dssims, dssim_f = build_model_and_evaluate_performance(i, j, name, only_data, modeltype=modeltype, metric=metric, feature=feature, json=json, testset=testset, featurelist=featurelist, xform=xform, jobid=jobid, labelsonly=True)
        if len(e) > 0:
            errors.append(e[0])
        if len(p) > 0:
            av_preds.append(p[0])
        if len(d) > 0:
            av_dssims.append(d[0])
        dssim_fs[i] = dssim_f


    if only_data or feature:
        exit()

    return errors, av_preds, av_dssims, test_dssims, dssim_fs
