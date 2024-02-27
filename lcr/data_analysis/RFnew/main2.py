import matplotlib.pyplot as plt
import numpy as np
from utils import parse_command_line_arguments, read_parameters_from_json
from model_training import build_and_evaluate_models_for_time_slices
import os
# os.environ["HDF5_PLUGIN_PATH"]
import datetime
from math import floor
from sklearn.metrics import confusion_matrix, classification_report
import os
import xarray as xr
from main import main
import argparse
import matplotlib
import csv
import seaborn as sns
import pandas as pd
from collections import Counter
matplotlib.use('Agg')

def find_first_true_cdir(truepass_dict, cdirs, i):
    first_true_cdirs = [None] * len(truepass_dict[cdirs[0]][i]['truepass']) # assuming 32 elements in the truepass array

    for idx in range(len(truepass_dict[cdirs[0]][i]['truepass'])):
        for cdir in cdirs:
            if truepass_dict[cdir][i]['truepass'][idx]:
                first_true_cdirs[idx] = cdir
                break

    return first_true_cdirs

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
    # ds = ds.set_index(collection=titles)

    ds.coords["collection"] = set_values
    ldcpy_dssims = ds.to_dataset(name='dssims')
    ldcpy_dssims.attrs['data_type'] = 'cam-fv'
    ds.coords["collection"] = titles

    ldcpy_dssims = ldcpy_dssims.set_coords('collection')
    ldcpy_dssims = ldcpy_dssims.set_index(collection='collection')

    # we need to add an index for the collection coordinates so that we can do ds.sel(collection="set1")

    # plot the dssims using matplotlib
    # plt.figure(figsize=(10, 5))
    # plt.imshow(ldcpy_dssims['dssims'].values.squeeze(), cmap='plasma')
    # plt.colorbar()
    # plt.title('DSSIMs')
    # plt.show()

    return ldcpy_dssims
def main2():

    WINDOWSIZE = 12

    # dataset = ldcpy.open_datasets(list_of_files=[opath + "TSxf
    truepred_dict = {}
    truedssim_dict = {}

    args = parse_command_line_arguments()

    j = args.json
    testset = args.testset
    featurelist = args.listfeatures
    xform = args.transform
    only_data = args.onlydata

    # only_data = args.onlydata
    model = args.model
    feature = args.feature
    jobid = args.jobid
    runonlydata = args.runonlydata
    labelsonly = args.labelsonly
    save, vlist, pre, post, opath, cpath, cdirs, ldcpypath, time, storageloc, n, stride, metric, cut_dataset, subdirs = read_parameters_from_json(j)
    fname = j.split(".")[0]


    # start by clearing the storage directory
    # os.system(f"rm -rf {storageloc}*")
    #
    if (runonlydata):
        main(metric_overwrite=metric, feature_override=True, newfeature=feature, only_data_override=True, newonlydata=True, j=j,
                testset=testset, featurelist=featurelist, xform=xform, jobid=jobid, model=model, feature=feature, only_data=only_data, labelsonly=labelsonly)
        # main(metric_overwrite="dssim", feature_override=True, newfeature="mean", only_data_override=True, newonlydata=True)
        # main(metric_overwrite="ks", feature_override=True, newfeature="mean", only_data_override=True, newonlydata=True)
        # main(metric_overwrite="pcc", feature_override=True, newfeature="mean", only_data_override=True, newonlydata=True)

    only_data = False
    feature = None
    main(metric_overwrite=metric, feature_override=True, newfeature=None, only_data_override=True, newonlydata=False, j=j,
                testset=testset, featurelist=featurelist, xform=xform, jobid=jobid, model=model, feature=None, only_data=only_data, labelsonly=labelsonly)
    # main(metric_overwrite="dssim", feature_override=True, newfeature=None, only_data_override=True, newonlydata=False)
    # main(metric_overwrite="ks", feature_override=True, newfeature=None, only_data_override=True, newonlydata=False)
    # main(metric_overwrite="pcc", feature_override=True, newfeature=None, only_data_override=True, newonlydata=False)

    if ldcpypath:
        import sys
    sys.path.append(ldcpypath)
    import ldcpy

    i = time[-1]

    # for metric in ["dssim", "spre", "ks", "pcc"]:
    for metric in [metric]:
        if labelsonly:
            # open the pred_f npy file containing a numpy array of predictions and the dssim_f npy file containing a numpy array of dssims
            # preds = np.load(pred_fs[cdir])
            for t in time:
                dssims = np.load(f"{storageloc}labels_{metric}_{fname}{t*len(subdirs)}{model}{jobid}_classify.npy", allow_pickle=True)
                fname = j.split(".")[0]
                preds = np.load(f"{storageloc}predictions_{metric}_{fname}{t*len(subdirs)}{model}{jobid}_classify.npy", allow_pickle=True)


            # for each time slice, compute whether the prediction is equal to or higher than the actual dssim
            # first, strip the top and bottom 5 rows from the dssims

            steps = int((i * len(subdirs)) * 0.75)
            # steps = int((i*len(subdirs)) * 0.2)

            # errs = preds - dssims[:,5:-5,steps[0]:]
            # # now if the value is greater than 0, set it to 1, otherwise set it to 0
            # truepass = np.where(errs > 0, 1, 0)
            # compute the average dssim over the entire time slice
        else:
            for cdir in cdirs:
                # open the pred_f npy file containing a numpy array of predictions and the dssim_f npy file containing a numpy array of dssims
                # preds = np.load(pred_fs[cdir])
                dssims = np.load(f"{storageloc}{cdir}_{metric}_mat_alltime_{i*len(subdirs)}_{fname}{jobid}.npy")
                fname = j.split(".")[0]
                preds = np.load(f"{storageloc}predictions_{metric}_{fname}{cdir}{i*len(subdirs)}{model}{jobid}.npy")

                # for each time slice, compute whether the prediction is equal to or higher than the actual dssim
                # first, strip the top and bottom 5 rows from the dssims

                steps = int((i*len(subdirs)) * 0.75)
                # steps = int((i*len(subdirs)) * 0.2)

                # errs = preds - dssims[:,5:-5,steps[0]:]
                # # now if the value is greater than 0, set it to 1, otherwise set it to 0
                # truepass = np.where(errs > 0, 1, 0)
                # compute the average dssim over the entire time slice
                adssims = np.mean(dssims[:, :, steps:], axis=(0, 1))
                # compute the average prediction over the entire time slice
                if cut_dataset:
                    apreds = np.mean(preds, axis=(0, 1))
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
            if not labelsonly:
                predresult[i] = find_first_true_cdir(truepred_dict, cdirs, i)
                dssimresult[i] = find_first_true_cdir(truedssim_dict, cdirs, i)
                classifyd = [element if element is not None else "None" for element in dssimresult[i]]
                classifyp = [element if element is not None else "None" for element in predresult[i]]
                cm = confusion_matrix(classifyd, classifyp, labels=list(set(classifyp + classifyd)))
                report = classification_report(classifyd, classifyp, labels=list(set(classifyp + classifyd)))
            else:
                classifyd = dssims[steps:]
                classifyp = preds
                cm = confusion_matrix(classifyd, classifyp, labels=list(set(np.append(classifyp, classifyd))))
                report = classification_report(classifyd, classifyp, labels=list(set(np.append(classifyp, classifyd))))

            # save the confusion matrix
            # also create a classification report

            date = datetime.datetime.now()
            date_string = date.strftime("%Y-%m-%d-%H-%M-%S")

            np.save(f"{storageloc}confusion_matrix_{metric}_{i}_{j.split('.')[0]}{jobid}{model}_{date_string}.npy", cm)
            with open(f"{storageloc}classification_report_{metric}_{i}_{j.split('.')[0]}{jobid}{model}_{date_string}.txt", 'w') as f:
                f.write(report)
            # fig = plt.figure()
            # plt.matshow(cm)
            # plt.title(f"Confusion Matrix for timesteps: {time}")
            # plt.colorbar()
            # plt.ylabel('True Label')
            # plt.xlabel('Predicated Label')
            # fig.savefig('confusion_matrix' + str(learning_values.pop()) + '.jpg')

        # Check if the file exists
        file_name = f"{storageloc}result_table_{metric}_{i}_{j.split('.')[0]}{jobid}{model}_{date_string}.csv"
        file_exists = os.path.isfile(file_name)

        # Open the file in append mode, create if it doesn't exist
        with open(file_name, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # If the file is new, write the header
            if not file_exists:
                writer.writerow(["vnames", "algorithms"])

            # Write the data
            writer.writerow([", ".join(vlist), ", ".join(classifyd)])

        print("Data written to", file_name)

        # Count the frequency of each element in classifyd
        frequency_dict = Counter(classifyd)

        # Ensure each compression level in cdirs is a key in the frequency_dict
        for compression_level in cdirs:
            if compression_level not in frequency_dict:
                frequency_dict[compression_level] = 0

        # Now frequency_dict has all the compression levels as keys, with their respective counts
        print(frequency_dict)

        df = pd.DataFrame([frequency_dict], columns=cdirs)

        # Insert the "name" column at the beginning of the DataFrame
        # Assuming vlist has a single value for this case, as there's only one row in df
        df.insert(0, 'name', vlist[0])

        # Write the DataFrame to a CSV file
        csv_file_path = 'compression_frequencies.csv'  # Specify your desired file path
        df.to_csv(csv_file_path, index=False)

        print(f"Data written to '{csv_file_path}' successfully.")

        # convert the strings to integers based on their index in cdirs
        predints = {}
        dssimints = {}
        if labelsonly:
            predints[i] = [cdirs.index(x) if x is not None and x != 'None' else len(cdirs) for x in classifyp]
            dssimints[i] = [cdirs.index(x) if x is not None and x != 'None' else len(cdirs) for x in classifyd]
        else:
            for i in time:
                predints[i] = [cdirs.index(x) if x is not None else len(cdirs) for x in predresult[i]]
                dssimints[i] = [cdirs.index(x) if x is not None else len(cdirs) for x in dssimresult[i]]

        # compute true positives ( if predresult - dssimresult = 0)
        # and true
        true_positives = {}
        false_positives = {}
        false_negatives = {}
        if labelsonly:
            for i in time:
                true_positives[i] = [1 if predints[i][j] == dssimints[i][j] else 0 for j in range(len(predints[i]))]
                false_positives[i] = [1 if predints[i][j] > dssimints[i][j] else 0 for j in range(len(predints[i]))]
                false_negatives[i] = [1 if predints[i][j] < dssimints[i][j] else 0 for j in range(len(predints[i]))]
        else:
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
            predresult[i] = ["Lossless" if x is None else x for x in classifyp]
            frequencies, bins = np.histogram(predresult[i], bins=np.arange(len(set(predresult[i])) + 1) - 0.5)
            unique_elements, frequencies = np.unique(predresult[i], return_counts=True)

            # Define colors for each unique label
            unique_labels = list(set(predresult[i]))
            colorints = np.array(predints[i]) - np.array(dssimints[i])
            colors = ['red' if x > 0 else 'green' if x < 0 else 'blue' for x in colorints]

            plt.clf()
            # get 80% of the total number of slices i
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

            plt.title(
                f"Predictions for {vliststring} with {nlevels} levels, \n windowed = {windowed}, total # of test slices: {nslices}",
                fontsize=18)

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

            plt.savefig(f"{storageloc}histogram_preds_{metric}_{i}_{j.split('.')[0]}{jobid}_{date_string}.png", bbox_inches='tight')
            plt.clf()

        # do the same for dssimresult[i]
        for i in time:
            # replace and potential Nones in predresult[i] with "None"
            predresult[i] = ["Lossless" if x is None else x for x in classifyp]
            unique_elements_pred, frequencies_pred = np.unique(predresult[i], return_counts=True)

            # Similarly for dssimresult[i]
            dssimresult[i] = ["Lossless" if x is None else x for x in classifyd]
            unique_elements_dssim, frequencies_dssim = np.unique(dssimresult[i], return_counts=True)

            # Define colors for each unique label
            unique_labels_pred = list(set(predresult[i]))
            colorints = np.array(predints[i]) - np.array(dssimints[i])
            colors_pred = ['red' if x > 0 else 'green' if x < 0 else 'blue' for x in colorints]

            # Define colors for each unique label
            unique_labels_dssim = list(set(dssimresult[i]))
            colorints = np.array(dssimints[i]) - np.array(dssimints[i])
            colors_dssim = ['red' if x > 0 else 'green' if x < 0 else 'blue' for x in colorints]

            df_pred = pd.DataFrame(
                {'Compression Level': unique_elements_pred, 'Frequency': frequencies_pred, 'Dataset': f'{model} Preds'})
            df_dssim = pd.DataFrame(
                {'Compression Level': unique_elements_dssim, 'Frequency': frequencies_dssim, 'Dataset': 'Actual'})

            df = pd.concat([df_pred, df_dssim])


            # Plot the histogram using plt.bar with individual colors
            sns.barplot(x='Compression Level', y='Frequency', hue='Dataset', data=df)

            # plt.xticks(bins[:-1], unique_labels, rotation=45)
            plt.title(f"Predictions for # of time slices: {i}")
            plt.xlabel("Compression Level")
            plt.ylabel("Frequency")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{storageloc}double_histogram_{metric}_{i}_{j.split('.')[0]}{jobid}_{date_string}.png", bbox_inches='tight')
            plt.clf()

        # let's combine the two histograms above into one

        if only_data or feature:
            exit()

        test_slices = [x - 1 for x in time]
        print(time)
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
                        if labelsonly:
                            fname = j.split(".")[0]
                            dssims = np.load(
                                f"{storageloc}labels_{metric}_{fname}{t * len(subdirs)}{model}{jobid}_classify.npy",
                                allow_pickle=True)
                            preds = np.load(
                                f"{storageloc}predictions_{metric}_{fname}{t * len(subdirs)}{model}{jobid}_classify.npy",
                                allow_pickle=True)
                        else:
                            dssims = np.load(f"{storageloc}labels_{metric}_{fname}{cdir}{t*len(subdirs)}{model}{jobid}.npy")
                            preds = np.load(f"{storageloc}predictions_{metric}_{fname}{cdir}{t*len(subdirs)}{model}{jobid}.npy")

                        # flips dssims and preds upside down
                        # dssims = np.flipud(dssims)
                        # test_dssims = np.flipud(test_dssims)
                        # preds = np.flipud(preds)
                        # pad the top and bottom of dssims and preds with 5 0s

                        dssims = np.pad(dssims, (((floor(WINDOWSIZE / 2)), (floor(WINDOWSIZE / 2))), (0, 0), (0, 0)),
                                        'constant', constant_values=0)
                        # test_dssims = np.pad(test_dssims, (((floor(WINDOWSIZE/2)), (floor(WINDOWSIZE/2))), (0, 0)), 'constant', constant_values=0)
                        preds = np.pad(preds, (((floor(WINDOWSIZE / 2)), (floor(WINDOWSIZE / 2))), (0, 0), (0, 0)), 'constant',
                                       constant_values=0)

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
                        allthings = convert_np_to_dssims([dssims, preds], [
                            f"Actual DSSIMs (windowed: {windowed}, {nslices} timesteps, {vliststring})", "Model Predictions"])
                        ldcpy.plot(allthings, "dssims", calc="mean",
                                   sets=[f"Actual DSSIMs (windowed: {windowed}, {nslices} timesteps, {vliststring})",
                                         "Model Predictions"],
                                   weighted=False, start=0, end=0, short_title=False, vert_plot=True,
                                   color="plasma", cmin=0, cmax=1)
                        plt.savefig(f"{storageloc}{vliststring}{cdir}_allthings{metric}_{t}_{name}_{date_string}_{model}_noerr.png",
                                    bbox_inches='tight')
                        plt.clf()

                        allthings = convert_np_to_dssims([dssims - preds],
                                                         [f"Error ({fname} {cdir} {t} {model})"])
                        ldcpy.plot(allthings, "dssims", calc="mean", sets=[f"Error ({fname} {cdir} {t} {model})"],
                                   weighted=False, start=0, end=0, short_title=True, vert_plot=True,
                                   color="PiYG")
                        plt.savefig(f"{storageloc}{vliststring}{cdir}_allthings{metric}_{t}_{name}_{date_string}_{model}_erroronly.png",
                                    bbox_inches='tight')
                        plt.clf()

                        try:
                            # add a plot of the original data, and the error between compressed and original.
                            allthings = convert_np_to_dssims([dssims, preds], [
                                f"Actual DSSIMs (windowed: {windowed}, {nslices} timesteps, {vliststring})", "Model Predictions"])
                            ldcpy.plot(allthings, "dssims", calc="mean",
                                       sets=[f"Actual DSSIMs (windowed: {windowed}, {nslices} timesteps, {vliststring})",
                                             "Model Predictions"],
                                       weighted=False, start=0, end=0, short_title=False, vert_plot=True,
                                       color="plasma", cmin=0, cmax=100)
                            plt.savefig(f"{storageloc}{vliststring}{cdir}_allthings{metric}_{t}_{name}_{date_string}_{model}_noerr_nomax.png",
                                        bbox_inches='tight')
                            plt.clf()

                            allthings_zoom = convert_np_to_dssims(
                                [np.log10(abs(1 - dssims) + 10 ** -10), np.log10(abs(1 - preds) + 10 ** -10)],
                                [f"Actual DSSIMs (log scale, windowed: {windowed}, {nslices} timesteps, {vliststring})",
                                 "Model Predictions"])
                            ldcpy.plot(allthings_zoom, "dssims", calc="mean", sets=[
                                f"Actual DSSIMs (log scale, windowed: {windowed}, {nslices} timesteps, {vliststring})",
                                "Model Predictions"],
                                       weighted=False, start=0, end=0, short_title=False, vert_plot=True,
                                       color="plasma", cmin=-10, cmax=0)
                            plt.savefig(f"{storageloc}{vliststring}{cdir}_allthings{metric}_zoomed_{t}_{name}_{date_string}_{model}_noerr.png",
                                        bbox_inches='tight')
                            plt.clf()

                            # add a plot with cmin as the min ddssim value
                            allthings_min = convert_np_to_dssims([dssims, preds], [
                                f"Actual DSSIMs (rescaled, windowed: {windowed}, {nslices} timesteps, {vliststring})",
                                "Model Predictions"])
                            ldcpy.plot(allthings_min, "dssims", calc="mean", sets=[
                                f"Actual DSSIMs (rescaled, windowed: {windowed}, {nslices} timesteps, {vliststring})",
                                "Model Predictions"],
                                       weighted=False, start=0, end=0, short_title=False, vert_plot=True,
                                       color="plasma", cmin=allthings_min.dssims[:, 6:-6, :, :].min().values.min(), cmax=1)
                            plt.savefig(f"{storageloc}{vliststring}{cdir}_allthings{metric}_min_{t}_{name}_{date_string}_{model}_noerr.png",
                                        bbox_inches='tight')
                            plt.clf()
                        except:
                            pass

                        # ldcpy.plot(dataset, "TS", calc="mean", sets=["labels_orig"],
                        #            weighted=False, start=t, end=t, short_title=True, vert_plot=True)
                        # plt.savefig(f"{storageloc}{cdir}_allthingsORIG_{t}_{name}.png", bbox_inches='tight')
                        # plt.clf()

                        # ldcpy.plot(dataset, "TS", calc="mean", sets=["labels_orig", "labels_comp"], calc_type="diff", weighted=False, start=t, end=t, short_title=True, vert_plot=True)
                        # plt.savefig(f"{storageloc}{cdir}_allthingsERRORS_{t}_{name}.png", bbox_inches='tight')
                        # plt.clf()

if __name__ == "__main__":
    main2()