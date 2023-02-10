# open a user-specified netCDF file and load it into an xarray dataset
# cut the xarray dataset into 11x11 lat-lon chunks
# compute the ssim_mat on the entire dataset, extract the dssim
# corresponding to each 11x11 square
# fit a CNN using the 11x11 chunk as training data and the dssim value as output
# average all the predictions to provide a DSSIM estimate for the entire dataset.

from sklearn.model_selection import train_test_split
import xarray as xr
import numpy as np
import os
import gc
import sys
import json
import typing
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import models

def actualsize(input_obj):
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

def open_dataset(filename: str) -> xr.Dataset:
    return xr.open_dataset(filename)

# A function that extracts every possible contiguous 11x11 lat-lon chunk from the dataset
def cut_dataset(dataset: xr.Dataset, time, varname, storageloc) -> xr.Dataset:
    if os.path.exists(f"{storageloc}{varname}_chunks_{time}.npy"):
        chunks = np.load(f"{storageloc}{varname}_chunks_{time}.npy")
        return chunks
    else:
        lat = dataset.lat
        lon = dataset.lon
        lat_vals = lat.values
        lon_vals = lon.values
        chunks = np.ndarray(shape=(50596, time, 11, 11))
        for i in range(len(lat_vals) - 10):
            for j in range(len(lon_vals) - 10):
                # extract the 11x11 lat-lon chunk
                chunk = dataset.sel(lat=slice(lat_vals[i], lat_vals[i+10]),
                                    lon=slice(lon_vals[j], lon_vals[j+10])).isel(time=slice(0, time))
                chunks[(i+1)*(j+1)-1] = chunk[varname].values[np.newaxis, :, :, :]


        # save chunks as an ndarray
        np.save(f"{storageloc}{varname}_chunks_{time}.npy", chunks)

        return chunks




# A function that computes the ldcpy function ssim_mat on the entire dataset, save the dssim matrix
# corresponding to each 11x11 square
def compute_ssim_mat(dataset: xr.Dataset) -> np.ndarray:
    ssim_mat = ldcpy.ssim_mat(dataset)
    dssim = ssim_mat.isel(time=0, window=0)
    return dssim

# fit a CNN using the 11x11 chunk as training data and the dssim value as output
# average all the predictions to provide a DSSIM estimate for the entire dataset.
# compare the DSSIM estimate to the actual DSSIM value
# return the average error
def fit_cnn(dataset: xr.Dataset, dssim: np.ndarray, time, varname, nvar, storageloc, testset="random") -> float:
    if os.path.exists(f"{storageloc}{varname}_model.h5"):
        model = models.load_model(f"{storageloc}{varname}_model.h5")
        with open(f"{storageloc}average_error.txt", "r") as f:
            average_error = float(f.read())
        return model, average_error
    else:
        for comp in dssim.keys():
            # use 90% of the data for training, 9% for validation, and 1% for testing
            if testset == "random":
                train_data, test_data, train_labels, test_labels = train_test_split(dataset[0:(50596*time*nvar)], dssim[comp], test_size=0.1)
                val_data, test_data, val_labels, test_labels = train_test_split(test_data, test_labels, test_size=0.1)
            elif testset == "oneout":
                train_data = dataset[0:(50596*time*nvar-2)]
                train_labels = dssim[comp][0:(50596*time*nvar-2)]
                val_data = dataset[(50596*time*nvar-2):(50596*time*nvar - 1)]
                val_labels = dssim[comp][(50596*time*nvar-2):(50596*time*nvar-1)]
                test_data = dataset[(50596*time*nvar-1):(50596*time*nvar)]
                test_labels = dssim[comp][(50596*time*nvar-1):(50596*time*nvar)]

            model = Sequential()
            model.add(Conv2D(32, (3, 3), input_shape=(11, 11, 1), name='conv1'))
            model.add(Activation('relu', name='relu1'))
            model.add(Conv2D(32, (3, 3), name='conv2'))
            model.add(Activation('relu', name='relu2'))
            model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool1'))
            model.add(Dropout(0.25, name='dropout1'))
            model.add(Flatten(name='flatten1'))
            model.add(Dense(512, name='dense1'))
            model.add(Activation('relu', name='relu3'))
            model.add(Dropout(0.5, name='dropout2'))
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

            model.fit(train_data, epochs=2, batch_size=32, validation_data=val_data)

            plt.plot(np.log10(model.history.history['loss']))
            plt.plot(np.log10(model.history.history['val_loss']))
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.show()

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

        return score[1], model


def read_jsonlist(metajson):
    comp = ""
    lev = []
    save = ""
    vlist = []
    pre = ""
    post = ""
    opath = ""
    cpath = ""
    cdirs = []
    ldcpypath = ""

    print("Reading jsonfile", metajson, " ...")
    if not os.path.exists(metajson):
        print("\n")
        print("*************************************************************************************")
        print("Warning: Specified json file does not exist: ", metajson)
        print("*************************************************************************************")
        print("\n")
    else:
        fd = open(metajson)
        metainfo = json.load(fd)
        if 'SaveDir' in metainfo:
            save = metainfo['SaveDir']
        if "VarList" in metainfo:
            vlist = metainfo['VarList']
        if "FilenamePre" in metainfo:
            pre = metainfo['FilenamePre']
        if "FilenamePost" in metainfo:
            post = metainfo['FilenamePost']
        if "OrigPath" in metainfo:
            opath = metainfo['OrigPath']
        if "CompPath" in metainfo:
            cpath = metainfo['CompPath']
        if "CompDirs" in metainfo:
            cdirs = metainfo['CompDirs']
        if "OptLdcpyDevPath" in metainfo:
            ldcpypath = metainfo['OptLdcpyDevPath']
        if "Time" in metainfo:
            time = metainfo['Time']
        if "StorageLoc" in metainfo:
            storage = metainfo['StorageLoc']

    return save, vlist, pre, post, opath, cpath, cdirs, ldcpypath, time, storage


# the main function runs open_dataset, cut_dataset, and compute_ssim_matd
def main():
    ### This version of the main function builds a separate CNN for each variable, useful for training to predict a single variable
    # read in the scratch.json configuration file that specifies the location of the datasets
    save, vlist, pre, post, opath, cpath, cdirs, ldcpypath, time, storageloc = read_jsonlist("CNN11_test.json")
    if ldcpypath:
        sys.path.insert(0, ldcpypath)
    import ldcpy

    # create a list of compressed paths by prepending cpath to each directory in cdirs
    cpaths = [cpath + cdir for cdir in cdirs]
    # add opath to the beginning of cpaths
    paths = [opath] + cpaths

    # check that dssim_mat, dssim_mat_1e_1, dssim_mat_1e_3, cut_dataset_orig exist in the current directory, and if so, load them

        # for every variable
    for varname in vlist:
        print(varname)
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
                dc = ldcpy.Diffcalcs(dataset_orig.isel(time=t), dataset_zfp.isel(time=t), data_type="cam-fv")
                dc.get_diff_calc("ssim_fp")
                dssim_mats[cdir][t] = dc._ssim_mat_fp[0].flatten()

        cut_dataset_orig = cut_dataset(dataset_col.sel(collection="orig"), time, varname, storageloc)
        # -1 means unspecified (should normally be 50596 * time unless
        # the number of time steps loaded by cut_dataset is different than time)
        cut_dataset_orig = cut_dataset_orig.reshape((-1, 11, 11))
        # flatten dssim_mats over time
        for i, cdir in enumerate(cdirs):
            dssim_mats[cdir] = dssim_mats[cdir].flatten()

        np.save(f"{storageloc}{varname}_dssim_mat_{time}.npy", dssim_mats)
        if not os.path.exists(f"{storageloc}{varname}_chunks_{time}.npy"):
            np.save(f"{storageloc}{varname}_chunks_{time}.npy", cut_dataset_orig)

            # hopefully, by this point dssim_mat contains all the dssims for a single variable at every compression level
            # and cut_dataset_orig contains all the uncompressed 11x11 chunks for a single variable

        # call fit_cnn on the 11x11 chunks and the dssim values
        error, model = fit_cnn(cut_dataset_orig, dssim_mats, time, varname, 1, storageloc)
    print(error)

def main1():
    ### This version of the main function builds a single CNN on all variables, useful for training to predict a new variable
    # read in the scratch.json configuration file that specifies the location of the datasets
    save, vlist, pre, post, opath, cpath, cdirs, ldcpypath, time, storageloc = read_jsonlist("CNN11_local.json")
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
                dc = ldcpy.Diffcalcs(dataset_orig.isel(time=t), dataset_zfp.isel(time=t), data_type="cam-fv")
                dc.get_diff_calc("ssim_fp")
                dssim_mats[cdir][t] = dc._ssim_mat_fp[0].flatten()

        cut_dataset_orig = cut_dataset(dataset_col.sel(collection="orig"), time, varname, storageloc)
        # -1 means unspecified (should normally be 50596 * time unless
        # the number of time steps loaded by cut_dataset is different than time)
        cut_dataset_orig = cut_dataset_orig.reshape((-1, 11, 11))
        # flatten dssim_mats over time
        for i, cdir in enumerate(cdirs):
            dssim_mats[cdir] = dssim_mats[cdir].flatten()

        np.save(f"{storageloc}{varname}_dssim_mat_{time}.npy", dssim_mats)
        if not os.path.exists(f"{storageloc}{varname}_chunks_{time}.npy"):
            np.save(f"{storageloc}{varname}_chunks_{time}.npy", cut_dataset_orig)

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
    error, model = fit_cnn(final_cut_dataset_orig, final_dssim_mats, time, "combine", len(vlist), storageloc, "oneout")
    print(error)


if __name__ == "__main__":
    main1()