import matplotlib.pyplot as plt
import numpy as np
# now import parseArguments from lcr/data_analysis/CNN11-new/utils.py and read_jsonlist from lcr/data_analysis/CNN11-new/utils.py
from utils import parse_command_line_arguments, read_parameters_from_json
from plotting import generate_performance_plots
from model_training import build_and_evaluate_models_for_time_slices
import os
# os.environ["HDF5_PLUGIN_PATH"]
import datetime
from math import floor
from sklearn.metrics import confusion_matrix
import os

WINDOWSIZE = 12


def convert_np_to_dssims(np_arrays, titles):
    das = []
    set_values = [f'set{i+1}' for i in range(len(np_arrays))]
    set_values = titles
    for i in range(len(np_arrays)):
        # first, flip the np array upside down
        # np_arrays[i] = np.flipud(np_arrays[i])
        # and roll it vertically by 10
        # stick the dssims and predictions into an xarray DataArray
        dssims_da = xr.DataArray(np_arrays[i])
        lat_values = np.linspace(-90, 90, np.shape(dssims_da)[0])
        lon_values = np.linspace(0, 360, np.shape(dssims_da)[1])
        # Rotate the longitude values by 180 degrees
        lon_values = np.where(lon_values > 180, lon_values - 360, lon_values)
        dssims_da.coords['latitude'] = (('dim_0'), lat_values)
        dssims_da.coords['longitude'] = (('dim_1'), lon_values)
        # Add new dimensions 'time' and 'sets'
        # Replace 'time_values' and 'set_values' with your actual time and set values
        if np.ndim(dssims_da) == 2:
            time_values = np.array([0])  # replace with your actual time values
            dssims_da = dssims_da.expand_dims({'time': time_values})
            dssims_da = dssims_da.rename({'dim_0': 'latitude', 'dim_1': 'longitude'})
        else:
            dssims_da = dssims_da.rename({'dim_0': 'latitude', 'dim_1': 'longitude', 'dim_2': 'time'})

        # Convert the DataArray to a Dataset
        # Assuming dssims_da is your DataArray and you've renamed the dimensions to 'latitude' and 'longitude'
        dssims_da['latitude'].attrs['units'] = 'degrees_north'
        dssims_da['longitude'].attrs['units'] = 'degrees_east'
        dssims_da.attrs['units'] = 'dssim'

        das.append(dssims_da)
    ds = xr.concat(das, dim=xr.DataArray(set_values, dims='collection'))
    ds.coords["collection"] = set_values
    ldcpy_dssims = ds.to_dataset(name='dssims')
    ldcpy_dssims.attrs['data_type'] = 'cam-fv'
    ds.coords["collection"] = titles

    # plot the dssims using matplotlib
    # plt.figure(figsize=(10, 5))
    # plt.imshow(ldcpy_dssims['dssims'].values.squeeze(), cmap='plasma')
    # plt.colorbar()
    # plt.title('DSSIMs')
    # plt.show()

    return ldcpy_dssims

def find_first_true_cdir(truepass_dict, cdirs, i):
    first_true_cdirs = [None] * len(truepass_dict[cdirs[0]][i]['truepass']) # assuming 32 elements in the truepass array

    for idx in range(len(truepass_dict[cdirs[0]][i]['truepass'])):
        for cdir in cdirs:
            if truepass_dict[cdir][i]['truepass'][idx]:
                first_true_cdirs[idx] = cdir
                break

    return first_true_cdirs

if __name__ == "__main__":
    args = parse_command_line_arguments()

    j = args.json
    only_data = args.onlydata
    model = args.model
    feature = args.feature
    jobid = args.jobid
    save, vlist, pre, post, opath, cpath, cdirs, ldcpypath, time, storageloc, n, stride, metric, cut_dataset = read_parameters_from_json(j)
    # times = [2, 3, 4]
    # n = 2

    # save the output of running p() 10 times to arrays
    errors_all = []
    av_preds_all = []
    av_dssims_all = []
    predictions_all = []
    errors1_all = []
    av_preds1_all = []
    av_dssims1_all = []
    errors3_all = []
    av_preds3_all = []
    av_dssims3_all = []

    if ldcpypath:
        import sys
        sys.path.append(ldcpypath)
        import ldcpy

    # dataset = ldcpy.open_datasets(list_of_files=[opath + "TSxf
    truepred_dict = {}
    truedssim_dict = {}


    for i in time:
        if feature or only_data:
            # j.split(".")[0] is the name of the json template
            build_and_evaluate_models_for_time_slices(time, j.split(".")[0], j.split(".")[0], only_data=only_data,
                                                      modeltype=model, feature=feature, metric=metric)
            break

        #here, we want to check if some files already exist. specifically:
        # f"{storageloc}predictions_{fname}{cdir}{i}{model}{jobid}.npy"
        # f"{storageloc}{cdir}_dssim_mat_alltime_{i}_{fname}{jobid}.npy"
        # f"{storageloc}labels_{fname}{cdir}{t}{model}{jobid}.npy"
        # f"{storageloc}predictions_{fname}{cdir}{i}{model}{jobid}.npy"
        fname = j.split(".")[0]
        f1 = f"{storageloc}predictions_{fname}{cdirs[0]}{i}{model}{jobid}.npy"
        f2 = f"{storageloc}{cdirs[0]}_dssim_mat_alltime_{i}_{fname}{jobid}.npy"
        f3 = f"{storageloc}labels_{fname}{cdirs[0]}{i}{model}{jobid}.npy"
        f4 = f"{storageloc}predictions_{fname}{cdirs[0]}{i}{model}{jobid}.npy"
        if not os.path.isfile(f1) or not os.path.isfile(f2) or not os.path.isfile(f3) or not os.path.isfile(f4):
            errors, av_preds, av_dssims, test_dssims, dssim_fs = build_and_evaluate_models_for_time_slices(time, j.split(".")[0], j.split(".")[0], only_data=only_data, modeltype=model, feature=feature, metric=metric)
        # errors_all.append(errors)
        # av_preds_all.append(av_preds)
        # av_dssims_all.append(av_dssims)
        # predictions_all.append(predictions)

        for cdir in cdirs:
            # open the pred_f npy file containing a numpy array of predictions and the dssim_f npy file containing a numpy array of dssims
            # preds = np.load(pred_fs[cdir])
            dssims = np.load(f"{storageloc}{cdir}_dssim_mat_alltime_{i}_{fname}{jobid}.npy")
            fname = j.split(".")[0]
            preds = np.load(f"{storageloc}predictions_{fname}{cdir}{i}{model}{jobid}.npy")

            # for each time slice, compute whether the prediction is equal to or higher than the actual dssim
            # first, strip the top and bottom 5 rows from the dssims
            steps = int(i * 0.2)
            # errs = preds - dssims[:,5:-5,steps[0]:]
            # # now if the value is greater than 0, set it to 1, otherwise set it to 0
            # truepass = np.where(errs > 0, 1, 0)
            # compute the average dssim over the entire time slice
            adssims = np.mean(dssims[:,:,steps:], axis=(0,1))
            # compute the average prediction over the entire time slice
            if cut_dataset:
                apreds = np.mean(preds, axis=(0,1))
            else:
                apreds = preds

            threshold = 0.995

            truepred = apreds > threshold
            if cdir not in truepred_dict:
                truepred_dict[cdir] = {}
            if i not in truepred_dict[cdir]:
                truepred_dict[cdir][i] = {}
            truepred_dict[cdir][i]['truepass'] = truepred

            # compute the average dssim over the entire time slice
            truedssim = adssims > threshold
            if cdir not in truedssim_dict:
                truedssim_dict[cdir] = {}
            if i not in truedssim_dict[cdir]:
                truedssim_dict[cdir][i] = {}
            truedssim_dict[cdir][i]['truepass'] = truedssim

    if only_data:
        exit()

    predresult = {}
    dssimresult = {}
    for i in time:
        predresult[i] = find_first_true_cdir(truepred_dict, cdirs, i)
        dssimresult[i] = find_first_true_cdir(truedssim_dict, cdirs, i)
        classifyd = [element if element is not None else "None" for element in dssimresult[i]]
        classifyp = [element if element is not None else "None" for element in predresult[i]]
        cm = confusion_matrix(classifyd, classifyp, labels=list(set(classifyp + classifyd)))
        # save the confusion matrix

        np.save(f"{storageloc}confusion_matrix_{i}_{j.split('.')[0]}{jobid}.npy", cm)
        # fig = plt.figure()
        # plt.matshow(cm)
        # plt.title(f"Confusion Matrix for timesteps: {time}")
        # plt.colorbar()
        # plt.ylabel('True Label')
        # plt.xlabel('Predicated Label')
        # fig.savefig('confusion_matrix' + str(learning_values.pop()) + '.jpg')

    # convert the strings to integers based on their index in cdirs
    predints = {}
    dssimints = {}
    for cdir in cdirs:
        for i in time:

            predints[i] = [cdirs.index(x) if x is not None else len(cdirs) for x in predresult[i]]
            dssimints[i] = [cdirs.index(x) if x is not None else len(cdirs) for x in dssimresult[i]]

    # compute true positives ( if predresult - dssimresult = 0)
    # and true
    true_positives = {}
    false_positives = {}
    false_negatives = {}
    for i in time:
        true_positives[i] = [1 if predints[i][j] == dssimints[i][j] else 0 for j in range(len(predresult[i]))]
        false_positives[i] = [1 if predints[i][j] > dssimints[i][j] else 0 for j in range(len(predresult[i]))]
        false_negatives[i] = [1 if predints[i][j] < dssimints[i][j] else 0 for j in range(len(predresult[i]))]

    # now sum up the true positives, false positives, and false negatives
    accuracy = {}
    false_negative_fraction = {}
    false_positive_fraction = {}
    for i in time:
        true_positives_sum = sum(true_positives[i])
        false_positives_sum = sum(false_positives[i])
        false_negatives_sum = sum(false_negatives[i])

        # compute accuracy as true_positives_sum / (true_positives_sum + false_positives_sum + false_negatives_sum)
        if (true_positives_sum + false_positives_sum + false_negatives_sum) > 0:
            accuracy[i] = true_positives_sum / (true_positives_sum + false_positives_sum + false_negatives_sum)
        else:
            accuracy[i] = 0
        if (true_positives_sum + false_negatives_sum) > 0:
            false_positive_fraction[i] = false_positives_sum / (true_positives_sum + false_negatives_sum)
        else:
            false_positive_fraction[i] = 0
        if (true_positives_sum + false_negatives_sum) > 0:
            false_negative_fraction[i] = false_negatives_sum / (true_positives_sum + false_negatives_sum)
        else:
            false_negative_fraction[i] = 0

    # build a histogram of predresult[i]

    for i in time:
        # replace and potential Nones in predresult[i] with "None"
        predresult[i] = ["Lossless" if x is None else x for x in predresult[i]]
        frequencies, bins = np.histogram(predresult[i], bins=np.arange(len(set(predresult[i])) + 1) - 0.5)
        unique_elements, frequencies = np.unique(predresult[i], return_counts=True)

        # Define colors for each unique label
        unique_labels = list(set(predresult[i]))
        colorints = np.array(predints[i]) - np.array(dssimints[i])
        colors = ['red' if x > 0 else 'green' if x < 0 else 'blue' for x in colorints]

        plt.clf()
        #get 80% of the total number of slices i
        nslices = int(i * 0.8)

        # Plot the histogram using plt.bar with individual colors
        # set the max height of the histogram to be 80% of the total number of slices
        plt.bar(bins[:-1], frequencies, color=colors, align='center', width=np.diff(bins))
        plt.ylim(0, nslices)
        plt.xticks(bins[:-1], unique_labels, rotation=45)
        # for each item in vlist, append to a string separated by a comma and space
        vliststring = ""
        for v in vlist:
            vliststring += v + ", "
        vliststring = vliststring[:-2]
        # for each cdir in cdirs, append to a string separated by a comma and space
        nlevels = len(cdirs)
        if cut_dataset:
            windowed = "true"
        else:
            windowed = "false"

        plt.title(f"Predictions for {vliststring} with {nlevels} levels, \n windowed = {windowed}, total # of test slices: {nslices}", fontsize=18)

        plt.tight_layout()
        # change plot font size to be much larger
        plt.rcParams.update({'font.size': 22})
        # change the text size of the x and y labels to be much larger
        plt.xlabel("Compression Level", fontsize=18)
        plt.ylabel("Frequency", fontsize=22)
        # also do this for the x and y tick labels
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=22)
        # can we shrink the plot size both horizontally and vertically?
        plt.gcf().subplots_adjust(bottom=0.3)
        plt.gcf().subplots_adjust(left=0.15)


        plt.show()



        plt.savefig(f"{storageloc}histogram_preds_{i}_{j.split('.')[0]}{jobid}.png", bbox_inches='tight')
        plt.clf()

    # do the same for dssimresult[i]
    for i in time:
        dssimresult[i] = ["Lossless" if x is None else x for x in dssimresult[i]]
        frequencies, bins = np.histogram(dssimresult[i], bins=np.arange(len(set(dssimresult[i])) + 1) - 0.5)
        unique_elements, frequencies = np.unique(predresult[i], return_counts=True)

        # Define colors for each unique label
        unique_labels = list(set(dssimresult[i]))
        colorints = np.array(dssimints[i]) - np.array(dssimints[i])
        colors = ['red' if x > 0 else 'green' if x < 0 else 'blue' for x in colorints]

        # Plot the histogram using plt.bar with individual colors
        plt.bar(bins[:-1], frequencies, color=colors, align='center', width=np.diff(bins))

        plt.xticks(bins[:-1], unique_labels, rotation=45)
        plt.title(f"Predictions for # of time slices: {i}")
        plt.xlabel("Compression Level")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"{storageloc}histogram_dssims_{i}_{j.split('.')[0]}{jobid}.png", bbox_inches='tight')
        plt.clf()

    if only_data or feature:
        exit()

    test_slices = [x - 1 for x in time]
    print (time)
    # print (errors_all)
    # generate_performance_plots(test_slices, [errors_all],
    #                   [av_dssims_all],
    #                   [av_preds_all],
    #                   cdirs, j.split(".")[0], save, j.split(".")[0], storageloc)
    import xarray as xr


    name = j.split(".")[0]
    for cdir in cdirs:
        print("Outside\n\n")
        if type(time) == list:
            print("Inside\n\n")
            for t in time:
                if cut_dataset:
                    date = datetime.datetime.now()
                    date_string = date.strftime("%Y-%m-%d-%H-%M-%S")
                    fname = j.split(".")[0]
                    # load the dssims and predictions
                    # dssims = np.load(f"{storageloc}{cdir}_dssim_mat_{t}_{name}.npy")
                    dssims = np.load(f"{storageloc}labels_{fname}{cdir}{t}{model}{jobid}.npy")
                    preds = np.load(f"{storageloc}predictions_{fname}{cdir}{t}{model}{jobid}.npy")

                    # flips dssims and preds upside down
                    # dssims = np.flipud(dssims)
                    # test_dssims = np.flipud(test_dssims)
                    # preds = np.flipud(preds)
                    # pad the top and bottom of dssims and preds with 5 0s


                    dssims = np.pad(dssims, (((floor(WINDOWSIZE/2)), (floor(WINDOWSIZE/2))), (0, 0), (0, 0)), 'constant', constant_values=0)
                    # test_dssims = np.pad(test_dssims, (((floor(WINDOWSIZE/2)), (floor(WINDOWSIZE/2))), (0, 0)), 'constant', constant_values=0)
                    preds = np.pad(preds, (((floor(WINDOWSIZE/2)), (floor(WINDOWSIZE/2))), (0, 0), (0, 0)), 'constant', constant_values=0)


                    ldcpy_dssims = convert_np_to_dssims([dssims], ["Actual DSSIMs"])
                    # ldcpy.plot(ldcpy_dssims, "dssims", calc="mean", sets=["Actual DSSIMs"], weighted=False, start=0, end=0, short_title=True, cmax=1, cmin=0, vert_plot=True)

                    # save the plots
                    # plt.savefig(f"{storageloc}{cdir}_dssim_mat_{t}_{name}.png", bbox_inches='tight')
                    # plt.clf()

                    da_preds = convert_np_to_dssims([preds], ["Model Predictions"])
                    # ldcpy.plot(da_preds, "dssims", calc="mean", sets=["Model Predictions"], weighted=False, start=0, end=0, short_title=True, cmax=1, cmin=0, vert_plot=True)


                    # save the plots
                    # plt.savefig(f"{storageloc}{cdir}_preds_{t}_{name}.png", bbox_inches='tight')

                    errors = [dssims - preds]
                    errors = convert_np_to_dssims(errors, ["Errors"])
                    # ldcpy.plot(errors, "dssims", calc="mean", sets=["Errors"], weighted=False, start=0, end=0, short_title=True, vert_plot=True)
                    # plt.savefig(f"{storageloc}{cdir}_error_{t}_{name}.png", bbox_inches='tight')
                    # plt.clf()

                    # FIX THESE NAMES

                    # allthings =convert_np_to_dssims([dssims, preds], [f"Actual DSSIM ({fname} {cdir} {t} {model})", f"Model Predictions ({fname} {cdir} {t} {model})"])
                    allthings =convert_np_to_dssims([dssims, preds], [f"Actual DSSIMs (windowed: {windowed}, {nslices} timesteps, {vliststring})", f"Model Predictions"])
                    ldcpy.plot(allthings, "dssims", calc="mean", sets=[f"Actual DSSIMs (windowed: {windowed}, {nslices} timesteps, {vliststring})", "Model Predictions"],
                               weighted=False, start=0, end=0, short_title=False, vert_plot=True,
                               color="plasma", cmin=0, cmax=1)
                    plt.savefig(f"{storageloc}{cdir}_allthingsDSSIMS_{t}_{name}_{date_string}_{model}_noerr.png", bbox_inches='tight')
                    plt.clf()



                    allthings = convert_np_to_dssims([dssims - preds],
                                                     [f"Error ({fname} {cdir} {t} {model})"])
                    ldcpy.plot(allthings, "dssims", calc="mean", sets=[f"Error ({fname} {cdir} {t} {model})"],
                               weighted=False, start=0, end=0, short_title=True, vert_plot=True,
                               color="PiYG")
                    plt.savefig(f"{storageloc}{cdir}_allthingsDSSIMS_{t}_{name}_{date_string}_{model}_erroronly.png", bbox_inches='tight')
                    plt.clf()

                    try:
                        allthings_zoom = convert_np_to_dssims([np.log10(abs(1-dssims)+10**-10), np.log10(abs(1-preds)+10**-10)], [f"Actual DSSIMs (windowed: {windowed}, {nslices} timesteps, {vliststring})", "Model Predictions"])
                        ldcpy.plot(allthings_zoom, "dssims", calc="mean", sets=[f"Actual DSSIMs (log scale, windowed: {windowed}, {nslices} timesteps, {vliststring})", "Model Predictions"],
                                   weighted=False, start=0, end=0, short_title=False, vert_plot=True,
                                   color="plasma", cmin=-10, cmax=0)
                        plt.savefig(f"{storageloc}{cdir}_allthingsDSSIMS_zoomed_{t}_{name}_{date_string}_{model}_noerr.png",
                                    bbox_inches='tight')
                        plt.clf()

                        # add a plot with cmin as the min ddssim value
                        allthings_min = convert_np_to_dssims([dssims, preds], [f"Actual DSSIMs (windowed: {windowed}, {nslices} timesteps, {vliststring})", "Model Predictions"])
                        ldcpy.plot(allthings_min, "dssims", calc="mean", sets=[f"Actual DSSIMs (rescaled, windowed: {windowed}, {nslices} timesteps, {vliststring})", "Model Predictions"],
                                   weighted=False, start=0, end=0, short_title=False, vert_plot=True,
                                   color="plasma", cmin=allthings_min.dssims[:,6:-6,:,:].min().values.min(), cmax=1)
                        plt.savefig(f"{storageloc}{cdir}_allthingsDSSIMS_min_{t}_{name}_{date_string}_{model}_noerr.png",
                                    bbox_inches='tight')
                        plt.clf()

                        # add a plot of the original data, and the error between compressed and original.


                    except:
                        pass


                    # ldcpy.plot(dataset, "TS", calc="mean", sets=["labels_orig"],
                    #            weighted=False, start=t, end=t, short_title=True, vert_plot=True)
                    # plt.savefig(f"{storageloc}{cdir}_allthingsORIG_{t}_{name}.png", bbox_inches='tight')
                    # plt.clf()

                    # ldcpy.plot(dataset, "TS", calc="mean", sets=["labels_orig", "labels_comp"], calc_type="diff", weighted=False, start=t, end=t, short_title=True, vert_plot=True)
                    # plt.savefig(f"{storageloc}{cdir}_allthingsERRORS_{t}_{name}.png", bbox_inches='tight')
                    # plt.clf()