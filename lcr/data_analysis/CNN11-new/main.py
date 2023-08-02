import matplotlib.pyplot as plt
import numpy as np
# now import parseArguments from lcr/data_analysis/CNN11-new/utils.py and read_jsonlist from lcr/data_analysis/CNN11-new/utils.py
from utils import parse_command_line_arguments, read_parameters_from_json
from plotting import generate_performance_plots
from model_training import build_and_evaluate_models_for_time_slices
import os
os.environ["HDF5_PLUGIN_PATH"]


def convert_np_to_dssims(np_arrays, titles):
    das = []
    set_values = [f'set{i+1}' for i in range(len(np_arrays))]
    set_values = titles
    for i in range(len(np_arrays)):
        # stick the dssims and predictions into an xarray DataArray
        dssims_da = xr.DataArray(np_arrays[i])
        lat_values = np.linspace(-90, 90, dssims.shape[0])
        lon_values = np.linspace(0, 360, dssims.shape[1])
        # Rotate the longitude values by 180 degrees
        lon_values = np.where(lon_values > 180, lon_values - 360, lon_values)
        dssims_da.coords['latitude'] = (('dim_0'), lat_values)
        dssims_da.coords['longitude'] = (('dim_1'), lon_values)
        # Add new dimensions 'time' and 'sets'
        # Replace 'time_values' and 'set_values' with your actual time and set values
        time_values = np.array([0])  # replace with your actual time values
        dssims_da = dssims_da.expand_dims({'time': time_values})
        # Convert the DataArray to a Dataset
        dssims_da = dssims_da.rename({'dim_0': 'latitude', 'dim_1': 'longitude'})
        # Assuming dssims_da is your DataArray and you've renamed the dimensions to 'latitude' and 'longitude'
        dssims_da['latitude'].attrs['units'] = 'degrees_north'
        dssims_da['longitude'].attrs['units'] = 'degrees_east'
        dssims_da.attrs['units'] = 'dssim'
        das.append(dssims_da)
    ds = xr.concat(das, dim=xr.DataArray(set_values, dims='collection'))
    ldcpy_dssims = ds.to_dataset(name='dssims')
    ldcpy_dssims.attrs['data_type'] = 'cam-fv'
    return ldcpy_dssims

if __name__ == "__main__":
    args = parse_command_line_arguments()

    j = args.json
    only_data = args.onlydata
    save, vlist, pre, post, opath, cpath, cdirs, ldcpypath, time, storageloc, n, stride = read_parameters_from_json(j)
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

    for i in range(n):
        errors, av_preds, av_dssims, predictions, test_dssims= build_and_evaluate_models_for_time_slices(time, j.split(".")[0], j.split(".")[0], only_data=False)
        if only_data:
            break
        errors_all.append(errors)
        av_preds_all.append(av_preds)
        av_dssims_all.append(av_dssims)
        predictions_all.append(predictions)
        # errors1_all.append(errors1)
        # av_preds1_all.append(av_preds1)
        # av_dssims1_all.append(av_dssims1)
        # errors3_all.append(errors3)
        # av_preds3_all.append(av_preds3)
        # av_dssims3_all.append(av_dssims3)
        # # save the predictions to a file
        # preds = np.zeros((182, 278)).flatten()
        # # set the values of mymap to the first 50596 values of predictions
        # if len(predictions) < 50596:
        #     preds[0:(len(predictions))] = predictions.squeeze()
        #     preds.reshape((182, 278))
        # else:
        #     preds = predictions.squeeze()[0:50596].reshape((182, 278))
        # # save the predictions to a file
        # np.save(f"{storageloc}predictions_{j.split('.')[0]}_{i}.npy", preds)
        # # also save the actual dssims

    if only_data:
        exit()

    test_slices = [x - 1 for x in time]
    print (time)
    print (errors_all)
    # generate_performance_plots(test_slices, [errors_all],
    #                   [av_dssims_all],
    #                   [av_preds_all],
    #                   cdirs, j.split(".")[0], save, j.split(".")[0], storageloc)
    import xarray as xr


    name = j.split(".")[0]
    for cdir in cdirs:
        if type(time) == list:
            for t in time:

                # load the dssims and predictions
                dssims = np.load(f"{storageloc}{cdir}_dssim_mat_{t}_{name}.npy")
                preds = np.load(f"{storageloc}{cdir}_preds_{t}_{name}.npy")
                # flips dssims and preds upside down
                dssims = np.flipud(dssims)
                test_dssims = np.flipud(test_dssims)
                preds = np.flipud(preds)
                # pad the top and bottom of dssims and preds with 5 0s
                dssims = np.pad(dssims, ((5, 5), (0, 0)), 'constant', constant_values=0)
                test_dssims = np.pad(test_dssims, ((5, 5), (0, 0)), 'constant', constant_values=0)
                preds = np.pad(preds, ((5, 5), (0, 0)), 'constant', constant_values=0)

                ldcpy_dssims = convert_np_to_dssims([test_dssims], ["Actual DSSIMs"])
                # ldcpy.plot(ldcpy_dssims, "dssims", calc="mean", sets=["Actual DSSIMs"], weighted=False, start=0, end=0, short_title=True, cmax=1, cmin=0, vert_plot=True)

                # save the plots
                # plt.savefig(f"{storageloc}{cdir}_dssim_mat_{t}_{name}.png", bbox_inches='tight')
                # plt.clf()

                da_preds = convert_np_to_dssims([preds], ["Model Predictions"])
                # ldcpy.plot(da_preds, "dssims", calc="mean", sets=["Model Predictions"], weighted=False, start=0, end=0, short_title=True, cmax=1, cmin=0, vert_plot=True)


                # save the plots
                # plt.savefig(f"{storageloc}{cdir}_preds_{t}_{name}.png", bbox_inches='tight')

                errors = [test_dssims - preds]
                errors = convert_np_to_dssims(errors, ["Errors"])
                # ldcpy.plot(errors, "dssims", calc="mean", sets=["Errors"], weighted=False, start=0, end=0, short_title=True, vert_plot=True)
                # plt.savefig(f"{storageloc}{cdir}_error_{t}_{name}.png", bbox_inches='tight')
                # plt.clf()

                allthings = convert_np_to_dssims([test_dssims, preds, test_dssims - preds], ["Actual DSSIMs", "Model Predictions", "Error"])
                ldcpy.plot(allthings, "dssims", calc="mean", sets=["Actual DSSIMs", "Model Predictions", "Error"], weighted=False, start=0, end=0, short_title=True, cmax=1, cmin=0, vert_plot=True, color="plasma")
                plt.savefig(f"{storageloc}{cdir}_allthingsDSSIMS_{t}_{name}.png", bbox_inches='tight')
                plt.clf()

                # ldcpy.plot(dataset, "TS", calc="mean", sets=["labels_orig"],
                #            weighted=False, start=t, end=t, short_title=True, vert_plot=True)
                # plt.savefig(f"{storageloc}{cdir}_allthingsORIG_{t}_{name}.png", bbox_inches='tight')
                # plt.clf()

                # ldcpy.plot(dataset, "TS", calc="mean", sets=["labels_orig", "labels_comp"], calc_type="diff", weighted=False, start=t, end=t, short_title=True, vert_plot=True)
                # plt.savefig(f"{storageloc}{cdir}_allthingsERRORS_{t}_{name}.png", bbox_inches='tight')
                # plt.clf()


        else:
            t = time
            # load the dssims and predictions
            dssims = np.load(f"{storageloc}{cdir}_dssim_mat_{t}_{name}.npy")
            preds = np.load(f"{storageloc}{cdir}_preds_{t}_{name}.npy")
            # flips dssims and preds upside down
            dssims = np.flipud(dssims)
            preds = np.flipud(preds)
            # roll the dssims and preds by 10

            ldcpy_dssims = convert_np_to_dssims([dssims], ["Actual DSSIMs"])
            ldcpy.plot(ldcpy_dssims, "dssims", calc="mean", sets=["Actual DSSIMs"], weighted=False, start=0, end=0,
                       short_title=True, cmax=1, cmin=0, vert_plot=True)

            # save the plots
            plt.savefig(f"{storageloc}{cdir}_dssim_mat_{t}_{name}.png", bbox_inches='tight')
            plt.clf()

            da_preds = convert_np_to_dssims([preds], ["Model Predictions"])
            ldcpy.plot(da_preds, "dssims", calc="mean", sets=["Model Predictions"], weighted=False, start=0, end=0,
                       short_title=True, cmax=1, cmin=0, vert_plot=True)

            # save the plots
            plt.savefig(f"{storageloc}{cdir}_preds_{t}_{name}.png", bbox_inches='tight')

            errors = [dssims - preds]
            errors = convert_np_to_dssims(errors, ["Errors"])
            ldcpy.plot(errors, "dssims", calc="mean", sets=["Errors"], weighted=False, start=0, end=0, short_title=True,
                       vert_plot=True)
            plt.savefig(f"{storageloc}{cdir}_error_{t}_{name}.png", bbox_inches='tight')
            plt.clf()

            allthings = convert_np_to_dssims([dssims, preds, dssims - preds],
                                             ["Actual DSSIMs", "Model Predictions", "Error"])
            ldcpy.plot(allthings, "dssims", calc="mean", sets=["Actual DSSIMs", "Model Predictions", "Error"],
                       weighted=False, start=0, end=0, short_title=True, cmax=1, cmin=0, vert_plot=True)
            plt.savefig(f"{storageloc}{cdir}_allthingsDSSIMS_{t}_{name}.png", bbox_inches='tight')
            plt.clf()

            ldcpy.plot(dataset, "TS", calc="mean", sets=["labels_orig"],
                       weighted=False, start=t, end=t, short_title=True, vert_plot=True)
            plt.savefig(f"{storageloc}{cdir}_allthingsORIG_{t}_{name}.png", bbox_inches='tight')
            plt.clf()

            ldcpy.plot(dataset, "TS", calc="mean", sets=["labels_orig", "labels_comp"], calc_type="diff",
                       weighted=False, start=t, end=t, short_title=True, vert_plot=True)
            plt.savefig(f"{storageloc}{cdir}_allthingsERRORS_{t}_{name}.png", bbox_inches='tight')
            plt.clf()

