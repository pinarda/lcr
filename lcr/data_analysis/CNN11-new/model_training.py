import os
import xarray as xr
import numpy as np
import sys
import tensorflow as tf
from utils import parse_command_line_arguments, read_parameters_from_json
from data_processing import cut_spatial_dataset_into_windows, split_data_into_train_val_test
from sklearn.preprocessing import quantile_transform
# import layers
# import random forest regressor
from sklearn.ensemble import RandomForestRegressor
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
        model = tf.keras.models.load_model(model_path)
        with open(average_error_path, "r") as f:
            average_error = float(f.read())
        return model, average_error
    else:
        scores = []
        av_preds = []
        av_dssims = []
        for comp in dssim.keys():
            # perform a quanitle transformation to make the data more Gaussian using the sklearn library
            for i in range(time*nvar):
                # dataset[i] = (dataset[i] - np.mean(dataset[i])) / np.std(dataset[i])
                train_data, train_labels, val_data, val_labels, test_data, test_labels = split_data_into_train_val_test(dataset, dssim, time, nvar,testset, comp)

            # model = Sequential()

            # let's try a deeper model.
            # model.add(Conv2D(16, (2, 2), input_shape=(11, 11, 1), name='conv1'))
            # model.add(Activation('relu', name='relu1'))
            # model.add(Conv2D(16, (2, 2), name='conv2'))
            # model.add(Activation('relu', name='relu2'))
            # model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool1'))
            # model.add(Dropout(0.25, name='dropout1'))
            # model.add(Flatten(name='flatten1'))
            # model.add(Dense(64, name='dense1'))
            # model.add(Activation('relu', name='relu5'))
            # model.add(Dropout(0.25, name='dropout3'))
            # model.add(Dense(1, name='dense2'))
            # model.add(Activation('linear', name='linear1'))
            # set the learning rate to 0.001

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
                        mse.append(float(line.split(",")[1]))
                    min_mse = min(mse)
                    min_mse_index = mse.index(min_mse)
                    filter1 = int(lines[min_mse_index].split(",")[2])
                    filter2 = int(lines[min_mse_index].split(",")[3])
                    dropout = float(lines[min_mse_index].split(",")[4])
                    batch_size = int(lines[min_mse_index].split(",")[5])
            else:
                filter1 = 16
                filter2 = 16
                dropout = 0.25
                batch_size = 32

            model = tf.keras.Sequential(
                [
                    tf.keras.Input(shape=(11, 11, 1)),
                    tf.keras.layers.Conv2D(filter1, kernel_size=(3, 3), activation="relu"),
                    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                    tf.keras.layers.Conv2D(filter2, kernel_size=(3, 3), activation="relu"),
                    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dropout(dropout),
                    tf.keras.layers.Dense(1, activation="linear"),
                ]
            )


            model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['mean_absolute_error'])

            model.summary()

            # use a random forest regressor
            # model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0)



            # apply a quantile transformation to each of the lat/lon slices of the data
            # this will make the data more Gaussian
            # first flatten the data
            train_data = train_data.reshape(train_data.shape[0], -1)
            val_data = val_data.reshape(val_data.shape[0], -1)
            test_data_OLD = test_data.reshape(test_data.shape[0], -1)
            train_data = quantile_transform(train_data, output_distribution='uniform', copy=True, n_quantiles=10000)
            val_data = quantile_transform(val_data, output_distribution='uniform', copy=True, n_quantiles=10000)
            test_data = quantile_transform(test_data_OLD, output_distribution='uniform', copy=True, n_quantiles=10000)
            # then put the data back into the original shape
            train_data = train_data.reshape(train_data.shape[0], 11, 11)
            val_data = val_data.reshape(val_data.shape[0], 11, 11)
            test_data = test_data.reshape(test_data.shape[0], 11, 11)
            test_data_OLD = test_data_OLD.reshape(test_data_OLD.shape[0], 11, 11)

            #plot the first 52416 values of test_labels in a 182*288 grid
            # make this the first of a 1x4 subplot
            import matplotlib.pyplot as plt
            plt.subplot(1, 2, 1)
            plt.imshow(test_labels[0:52416].reshape(182,288))
            test_plot = test_labels[0:52416].reshape(182,288)
            plt.show()

            # fit the model
            history = model.fit(train_data, train_labels, epochs=10, batch_size=batch_size, validation_data=(val_data, val_labels))

            score = model.evaluate(test_data, verbose=0)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])

            predictions = model.predict(test_data)
            plt.subplot(1, 2, 2)
            plt.imshow(predictions[0:52416].reshape(182, 288, order='C'))
            plt.show()
            average_prediction = np.mean(predictions)
            average_dssim = np.mean(test_labels)
            print("Average prediction: ", average_prediction)
            print("Average dssim: ", average_dssim)

            plt.subplot(2,3, 1)

            plt.imshow(test_labels[0:52416].reshape(182,288))
            plt.title(f"Test labels for {comp}")
            plt.subplot(2,3, 2)
            plt.imshow(predictions[0:52416].reshape(182, 288, order='C'))
            plt.title(f"Predictions for {comp}")
            plt.subplot(2,3, 3)
            # plot the errors (preds - labels)
            plt.imshow((predictions[0:52416].reshape(182, 288, order='C') - test_labels[0:52416].reshape(182,288)))
            plt.title(f"Errors for {comp}")
            plt.subplot(2,3, 4)
            # plot the test_data (only the center value of each 11x11 window)
            # change the color map for this subplot


            plt.imshow(test_data[0:52416,5,5].reshape(182, 288, order='C'), cmap='coolwarm')
            # title the subplot
            plt.title(f"Test data ({comp}, QT)")

            plt.subplot(2,3, 5)
            # plot the test_data (only the center value of each 11x11 window)
            plt.imshow(test_data_OLD[0:52416,5,5].reshape(182, 288, order='C'), cmap='coolwarm')
            # title the subplot
            plt.title(f"Train data ({comp})")
            # enlarge the figure so the subplots are not so close to each other
            plt.gcf().set_size_inches(20, 10)


            plt.savefig(f"{plotdir}{comp}_{j}.png")


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
        return scores, model, av_preds, av_dssims, predictions, test_plot


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
    # This version of the main function builds a single CNN on all variables, useful for training to predict a new variable
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
        # pad the longitude dimension of the original dataset by 5 on each side (wrap around)
        dataset_orig = xr.concat([dataset_orig[:, :, -5:], dataset_orig, dataset_orig[:, :, :5]], dim="lon")
        dssim_mats = {}
        for cdir in cdirs:
            dataset_zfp = dataset_col.sel(collection=cdir).to_array().squeeze()
            # pad the longitude dimension of the compressed dataset by 5 on each side (wrap around)
            dataset_zfp = xr.concat([dataset_zfp[:, :, -5:], dataset_zfp, dataset_zfp[:, :, :5]], dim="lon")
            dssim_mats[cdir] = np.empty((time, 52416))
            for t in range(0, time):
                dc = ldcpy.Diffcalcs(dataset_orig.isel(time=t*stride), dataset_zfp.isel(time=t*stride), data_type="cam-fv")
                dc.get_diff_calc("ssim_fp")
                dssim_mats[cdir][t] = dc._ssim_mat_fp[0].flatten()

        cut_dataset_orig = cut_spatial_dataset_into_windows(dataset_col.sel(collection="orig"), time, varname, storageloc)
        # -1 means unspecified (should normally be 50596 * time unless
        # the number of time steps loaded by cut_dataset is different than time)
        cut_dataset_orig = cut_dataset_orig.reshape((-1, 11, 11), order="F")
        # flatten dssim_mats over time
        for i, cdir in enumerate(cdirs):
            dssim_mats[cdir] = dssim_mats[cdir].flatten()

        np.save(f"{storageloc}{varname}_dssim_mat_{time}_{j}.npy", dssim_mats)
        # if not os.path.exists(f"{storageloc}{varname}_chunks_{time}.npy"):
        #     np.save(f"{storageloc}{varname}_chunks_{time}.npy", cut_dataset_orig)

            # hopefully, by this point dssim_mat contains all the dssims for a single variable at every compression level
            # and cut_dataset_orig contains all the uncompressed 11x11 chunks for a single variable
        #append cut_dataset_orig to final_cut_dataset_orig
        final_cut_dataset_orig = np.append(final_cut_dataset_orig, cut_dataset_orig[0:(52416*time)], axis=0)
        #append dssim_mats to final_dssim_mats
        for cdir in cdirs:
            if cdir not in final_dssim_mats:
                final_dssim_mats[cdir] = dssim_mats[cdir]
            else:
                final_dssim_mats[cdir] = np.append(final_dssim_mats[cdir], dssim_mats[cdir], axis=0)

    # call fit_cnn on the 11x11 chunks and the dssim values
    errors, model, av_preds, av_dssims, predictions, test_dssims = train_cnn_for_dssim_regression(final_cut_dataset_orig, final_dssim_mats, time, "combine", len(vlist), storageloc, testset, j)
    print(errors)
    # grab the first 50596 dssims for each compression level from dssim_mats
    # dssim_mats = {cdir: dssim_mats[cdir][0:50596] for cdir in cdirs}
    for cdir in cdirs:
        if type(time) is list:
            for t in time:
                np.save(f"{storageloc}{cdir}_dssim_mat_{t}_{name}.npy", final_dssim_mats[cdir][0:52416].reshape((182, 288)))
                # also save the predictions
                # and the errors
                preds = np.zeros((182, 288)).flatten()
                # set the values of mymap to the first 50596 values of predictions
                if len(predictions) < 52416:
                    preds[0:(len(predictions))] = predictions.squeeze()
                    preds = preds.reshape((182, 288))
                else:
                    preds = predictions.squeeze()[0:52416].reshape((182, 288))
                np.save(f"{storageloc}{cdir}_preds_{t}_{name}.npy", preds)
        else:
            np.save(f"{storageloc}{cdir}_dssim_mat_{time}_{name}.npy", final_dssim_mats[cdir][0:52416].reshape((182, 288)))
            # also save the predictions
            # and the errors
            preds = np.zeros((182, 288)).flatten()
            # set the values of mymap to the first 50596 values of predictions
            if len(predictions) < 52416:
                preds[0:(len(predictions))] = predictions.squeeze()
                preds = preds.reshape((182, 288))
            else:
                preds = predictions.squeeze()[0:52416].reshape((182, 288))
            np.save(f"{storageloc}{cdir}_preds_{time}_{name}.npy", preds)
    return errors, av_preds, av_dssims, predictions, test_dssims

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
        e, p, d, predictions, test_dssims = build_model_and_evaluate_performance(i, j, name)
        errors.append(e[0])
        av_preds.append(p[0])
        av_dssims.append(d[0])

    return errors, av_preds, av_dssims, predictions, test_dssims
