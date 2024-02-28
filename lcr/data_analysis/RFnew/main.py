import matplotlib.pyplot as plt
import numpy as np
# now import parseArguments from lcr/data_analysis/CNN11-new/utils.py and read_jsonlist from lcr/data_analysis/CNN11-new/utils.py
from utils import parse_command_line_arguments, read_parameters_from_json
from plotting import generate_performance_plots
from model_training import build_and_evaluate_models_for_time_slices
import os
os.environ["HDF5_PLUGIN_PATH"]
import datetime
from math import floor
from sklearn.metrics import confusion_matrix, classification_report
import os
import xarray as xr

WINDOWSIZE = 12






def main(metric_overwrite="dssim", feature_override=False, newfeature=None, only_data_override=False, newonlydata=False,
         j=None, testset="random",
         featurelist=None, xform="quantile", jobid=0, model="rf", feature=None, only_data=False, labelsonly=False):
    # args = parse_command_line_arguments()

    # j = args.json
    # only_data = args.onlydata
    # model = args.model
    # feature = args.feature
    # jobid = args.jobid

    save, vlist, pre, post, opath, cpath, cdirs, ldcpypath, time, storageloc, n, stride, metric, cut_dataset, subdirs = read_parameters_from_json(
        j)
    # times = [2, 3, 4]
    # n = 2
    if metric_overwrite:
        metric = metric_overwrite
    if feature_override:
        feature = newfeature
    if only_data_override:
        only_data = newonlydata


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

    for i in time:
        if feature or only_data:
            # j.split(".")[0] is the name of the json template
            build_and_evaluate_models_for_time_slices(time, j.split(".")[0], j.split(".")[0], only_data=only_data,
                                                      modeltype=model, feature=feature, metric=metric, json=j, testset=testset,
                                                featurelist=featurelist, xform=xform, jobid=jobid, labelsonly=labelsonly)
            break

    # here, we want to check if some files already exist. specifically:
    # f"{storageloc}predictions_{fname}{cdir}{i}{model}{jobid}.npy"
    # f"{storageloc}{cdir}_dssim_mat_alltime_{i}_{fname}{jobid}.npy"
    # f"{storageloc}labels_{fname}{cdir}{t}{model}{jobid}.npy"
    # f"{storageloc}predictions_{fname}{cdir}{i}{model}{jobid}.npy"
    fname = j.split(".")[0]
    f1 = f"{storageloc}predictions_{metric}_{fname}{cdirs[0]}{i}{model}{jobid}.npy"
    f2 = f"{storageloc}{cdirs[0]}_{metric}_mat_alltime_{i}_{fname}{jobid}.npy"
    f3 = f"{storageloc}labels_{metric}_{fname}{cdirs[0]}{i}{model}{jobid}.npy"
    f4 = f"{storageloc}predictions_{metric}_{fname}{cdirs[0]}{i}{model}{jobid}.npy"
    if not os.path.isfile(f1) or not os.path.isfile(f2) or not os.path.isfile(f3) or not os.path.isfile(f4):
        errors, av_preds, av_dssims, test_dssims, dssim_fs = build_and_evaluate_models_for_time_slices(time,
                                                                                                       j.split(".")[0],
                                                                                                       j.split(".")[0],
                                                                                                       only_data=only_data,
                                                                                                       modeltype=model,
                                                                                                       feature=feature,
                                                                                                       metric=metric,
                                                                                                       json=j,
                                                                                                       testset=testset,
                                                                                                       featurelist=featurelist,
                                                                                                       xform=xform,
                                                                                                       jobid=jobid,
                                                                                                       labelsonly=labelsonly)

if __name__ == "__main__":
    main()