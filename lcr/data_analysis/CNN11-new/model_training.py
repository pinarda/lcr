import os
import xarray as xr
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import models
from utils import parse_command_line_arguments, read_parameters_from_json
from data_processing import cut_spatial_dataset_into_windows, split_data_into_train_val_test
os.environ["HDF5_PLUGIN_PATH"]

def train_cnn_for_dssim_regression(dataset: xr.Dataset, dssim: np.ndarray, time, varname, nvar, storageloc,
                                   testset="random", j=None, plotdir=None, window_size=11) -> float:
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
    model_path = os.path.join(storageloc, f"{varname}_model.h5")
    average_error_path = os.path.join(storageloc, "average_error.txt")

    if os.path.exists(model_path):
        model = models.load_model(model_path)
        with open(average_error_path, "r") as f:
            average_error = float(f.read())
        return model, average_error
    else:
        scores = []
        av_preds = []
        av_dssims = []
        for comp in dssim.keys():
            # begin by standardizing all data to have mean 0 and standard deviation 1
            # this is done by subtracting the mean and dividing bdatay the standard deviation
            # of the training data
            for i in range(time*nvar):
                dataset[i] = (dataset[i] - np.mean(dataset[i])) / np.std(dataset[i])
                train_data, train_labels, val_data, val_labels, test_data, test_labels = split_data_into_train_val_test(dataset, dssim, time, nvar,testset, comp)

            model = Sequential()

            # let's try a deeper model.
            model.add(Conv2D(16, (3, 3), input_shape=(11, 11, 1), name='conv1'))
            model.add(Activation('relu', name='relu1'))
            model.add(Conv2D(16, (3, 3), name='conv2'))
            model.add(Activation('relu', name='relu2'))
            model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool1'))
            model.add(Dropout(0.25, name='dropout1'))
            model.add(Conv2D(32, (3, 3), name='conv3'))
            model.add(Activation('relu', name='relu3'))
            model.add(Flatten(name='flatten1'))
            model.add(Dense(64, name='dense1'))
            model.add(Activation('relu', name='relu5'))
            model.add(Dropout(0.25, name='dropout3'))
            model.add(Dense(1, name='dense2'))
            model.add(Activation('linear', name='linear1'))


            model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

            model.summary()

            train_data = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
            val_data = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
            test_data = tf.data.Dataset.from_tensor_slices((test_data, test_labels))

            train_data = train_data.map(lambda x, y: (tf.reshape(x, (11, 11, 1)), y)).batch(32).shuffle(10000).prefetch(1)
            val_data = val_data.map(lambda x, y: (tf.reshape(x, (11, 11, 1)), y)).batch(32).shuffle(10000).prefetch(1)
            test_data = test_data.map(lambda x, y: (tf.reshape(x, (11, 11, 1)), y)).batch(32).shuffle(10000).prefetch(1)

            # model.fit(train_data, epochs=2, batch_size=32, validation_data=val_data)
            model.fit(train_data, epochs=2, batch_size=32)

            score = model.evaluate(test_data, verbose=0)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])

            predictions = model.predict(test_data)
            average_prediction = np.mean(predictions)
            average_dssim = np.mean(test_labels)
            print("Average prediction: ", average_prediction)
            print("Average dssim: ", average_dssim)

            #model.save("model.h5")
            with open(f"{storageloc}average_error.txt", "w") as f:
                f.write(str(score[1]))

            # classification stuff
            # run_classification(predictions, test_labels)

            scores.append(score[1])
            av_preds.append(average_prediction)
            av_dssims.append(average_dssim)

            # save the model
            model.save(f"model_{j}.h5")
        return scores, model, av_preds, av_dssims, predictions


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


def build_model_and_evaluate_performance(timeoverride=None, j=0, name="", stride=1):
    args = parse_command_line_arguments()

    json = args.json
    testset = args.testset
    ### This version of the main function builds a single CNN on all variables, useful for training to predict a new variable
    # read in the scratch.json configuration file that specifies the location of the datasets
    save, vlist, pre, post, opath, cpath, cdirs, ldcpypath, time, storageloc, navg, stride = read_parameters_from_json(json)
    if timeoverride is not None:
        time = timeoverride
    if ldcpypath:
        sys.path.insert(0, ldcpypath)
    import ldcpy
    # create a list of compressed paths by prepending cpath to each directory in cdirs
    cpaths = [cpath + cdir for cdir in cdirs]
    # add opath to the beginning of cpaths
    paths = [opath] + cpaths

    # check that dssim_mat, dssim_mat_1e_1, dssim_mat_1e_3, cut_dataset_orig exist in the current directory, and if so, load them

    # for every variable
    final_cut_dataset_orig = np.empty((0, 11, 11))
    final_dssim_mats = {}
    for varname in vlist:
        files = []
        for path in paths:
            # create a list of files to open by appending pre, varname, and post to the filename
            # for every variable in vlist
            files.append(f"{path}/" + pre + varname + post)

            # create a list of labels by adding "orig" to the list of cdirs
        labels = ["orig"] + cdirs
        dataset_col = ldcpy.open_datasets(list_of_files=files,
                                          labels=labels,
                                          data_type="cam-fv",
                                          varnames=[varname])

        # extract the original and compressed dataarrays
        dataset_orig = dataset_col.sel(collection="orig").to_array().squeeze()
        dssim_mats = {}
        for cdir in cdirs:
            dataset_zfp = dataset_col.sel(collection=cdir).to_array().squeeze()
            dssim_mats[cdir] = np.empty((time, 50596))
            for t in range(0, time):
                dc = ldcpy.Diffcalcs(dataset_orig.isel(time=t*stride), dataset_zfp.isel(time=t*stride), data_type="cam-fv")
                dc.get_diff_calc("ssim_fp")
                dssim_mats[cdir][t] = dc._ssim_mat_fp[0].flatten()

        cut_dataset_orig = cut_spatial_dataset_into_windows(dataset_col.sel(collection="orig"), time, varname, storageloc)
        # -1 means unspecified (should normally be 50596 * time unless
        # the number of time steps loaded by cut_dataset is different than time)
        cut_dataset_orig = cut_dataset_orig.reshape((-1, 11, 11))
        # flatten dssim_mats over time
        for i, cdir in enumerate(cdirs):
            dssim_mats[cdir] = dssim_mats[cdir].flatten()

        np.save(f"{storageloc}{varname}_dssim_mat_{time}_{j}.npy", dssim_mats)
        # if not os.path.exists(f"{storageloc}{varname}_chunks_{time}.npy"):
        #     np.save(f"{storageloc}{varname}_chunks_{time}.npy", cut_dataset_orig)

            # hopefully, by this point dssim_mat contains all the dssims for a single variable at every compression level
            # and cut_dataset_orig contains all the uncompressed 11x11 chunks for a single variable
        #append cut_dataset_orig to final_cut_dataset_orig
        final_cut_dataset_orig = np.append(final_cut_dataset_orig, cut_dataset_orig[0:(50596*time)], axis=0)
        #append dssim_mats to final_dssim_mats
        for cdir in cdirs:
            if cdir not in final_dssim_mats:
                final_dssim_mats[cdir] = dssim_mats[cdir]
            else:
                final_dssim_mats[cdir] = np.append(final_dssim_mats[cdir], dssim_mats[cdir], axis=0)

    # call fit_cnn on the 11x11 chunks and the dssim values
    errors, model, av_preds, av_dssims, predictions = train_cnn_for_dssim_regression(final_cut_dataset_orig, final_dssim_mats, time, "combine", len(vlist), storageloc, testset, j)
    print(errors)
    # grab the first 50596 dssims for each compression level from dssim_mats
    # dssim_mats = {cdir: dssim_mats[cdir][0:50596] for cdir in cdirs}
    for cdir in cdirs:
        if type(time) is list:
            for t in time:
                np.save(f"{storageloc}{cdir}_dssim_mat_{t}_{name}.npy", final_dssim_mats[cdir][0:50596].reshape((182, 278)))
                # also save the predictions
                # and the errors
                preds = np.zeros((182, 278)).flatten()
                # set the values of mymap to the first 50596 values of predictions
                if len(predictions) < 50596:
                    preds[0:(len(predictions))] = predictions.squeeze()
                    preds = preds.reshape((182, 278))
                else:
                    preds = predictions.squeeze()[0:50596].reshape((182, 278))
                np.save(f"{storageloc}{cdir}_preds_{t}_{name}.npy", preds)
        else:
            np.save(f"{storageloc}{cdir}_dssim_mat_{time}_{name}.npy", final_dssim_mats[cdir][0:50596].reshape((182, 278)))
            # also save the predictions
            # and the errors
            preds = np.zeros((182, 278)).flatten()
            # set the values of mymap to the first 50596 values of predictions
            if len(predictions) < 50596:
                preds[0:(len(predictions))] = predictions.squeeze()
                preds = preds.reshape((182, 278))
            else:
                preds = predictions.squeeze()[0:50596].reshape((182, 278))
            np.save(f"{storageloc}{cdir}_preds_{time}_{name}.npy", preds)
    return errors, av_preds, av_dssims, predictions

def build_and_evaluate_models_for_time_slices(times, j, name):
    # Will need to perform a normalization step.
    errors = []
    av_preds = []
    av_dssims = []

    errors1 = []
    av_preds1 = []
    av_dssims1 = []

    errors3 = []
    av_preds3 = []
    av_dssims3 = []

    for i in times:
        e, p, d, predictions = build_model_and_evaluate_performance(i, j, name)
        errors.append(e[0])
        av_preds.append(p[0])
        av_dssims.append(d[0])

        errors1.append(e[1])
        av_preds1.append(p[1])
        av_dssims1.append(d[1])

        errors3.append(e[2])
        av_preds3.append(p[2])
        av_dssims3.append(d[2])
    return errors, av_preds, av_dssims, predictions, errors1, av_preds1, av_dssims1, errors3, av_preds3, av_dssims3
