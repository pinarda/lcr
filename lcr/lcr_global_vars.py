import re
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
            m = re.search('^(?P<compression>.*?)_(?P<level>[0-9]+?)_(?P<varname>.*)$', row[0])
            if m is not None:
                vars.append(m.group("varname"))
    vars = np.unique(vars)
    return vars

# manually specified variable option
# monthly_vars = ["FLNS", "FLNT", "FSNS", "FSNT", "LHFLX",
#              "PRECC", "PRECL", "PS", "QFLX", "SHFLX", "TMQ", "TS"]

# daily_vars = ["FLUT", "LHFLX", "PRECT", "TAUX", "TS", "Z500"]

# variables automatically specified based on available dssim calcs
#monthly_vars = varlist(f"../data/monthly_dssims.csv")
#daily_vars = varlist(f"../data/daily_dssims.csv")
daily_vars = ["bc_a1_SRF", "dst_a1_SRF", "dst_a3_SRF", "FLNS", "FLNSC",
               "FLUT", "FSNS", "FSNSC", "FSNTOA", "ICEFRAC", "LHFLX", "pom_a1_SRF", "PRECL", "PRECSC",
               "PRECSL", "PRECT", "PRECTMX", "PSL", "Q200", "Q500", "Q850", "QBOT", "SHFLX", "so4_a1_SRF",
               "so4_a2_SRF", "so4_a3_SRF", "soa_a1_SRF", "soa_a2_SRF", "T010", "T200", "T500", "T850",
               "TAUX", "TAUY", "TMQ", "TREFHT", "TREFHTMN", "TREFHTMX", "TS", "U010", "U200", "U500", "U850", "VBOT",
               "WSPDSRFAV", "Z050", "Z500"]

# array of the name prefixes for each compression algorithm
# used in most csv files and most other places in the python code
alg_prefixes = ["zfp", "bg", "sz", "sz1ROn", "zfp5", "sz3_ROn"]
# array of the names of each compression algorothm
# used in hist_plotter as x-axis, dataframe in create_dataframe.py
# also monthly_labels.csv and daily_labels.csv
algs = ["zfp", "bg", "sz", "sz1413", "z_hdf5", "sz3"]

# list of single-value features being used by the models
features = ["mean", "variance", "ns_con_var", "w_e_first_differences", "prob_positive", "num_zero", "range", "quantile"]