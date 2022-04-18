import regex as re
import csv
import numpy as np

def varlist(csvfilename):
    """
    Gets list of variables in a csv file
    """
    vars = []
    with open(csvfilename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 0:
                continue
            m = re.search('.*_(?P<level>.+)_(?P<varname>.*)', row[0])
            if m is not None:
                vars.append(m.group("varname"))
    vars = np.unique(vars)
    return vars

# manually specified variable option
# monthly_vars = ["FLNS", "FLNT", "FSNS", "FSNT", "LHFLX",
#              "PRECC", "PRECL", "PS", "QFLX", "SHFLX", "TMQ", "TS"]

# daily_vars = ["FLUT", "LHFLX", "PRECT", "TAUX", "TS", "Z500"]

# variables automatically specified based on available dssim calcs
monthly_vars = varlist(f"../data/test_set/monthly_dssims.csv")
daily_vars = varlist(f"../data/test_set/daily_dssims.csv")

# array of the name prefixes for each compression algorithm
# used in most csv files and most other places in the python code
alg_prefixes = ["zfp", "bg", "sz", "sz1ROn", "zfp5"]
# array of the names of each compression algorothm
# used in hist_plotter as x-axis, dataframe in create_dataframe.py
# also monthly_labels.csv and daily_labels.csv
algs = ["zfp", "bg", "sz", "sz1413", "z_hdf5"]

# list of single-value features being used by the models
features = ["mean", "variance", "ns_con_var"]