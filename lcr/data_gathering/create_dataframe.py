"""
This script requires daily_calcs.csv, monthly_calcs.csv (from DataGathering.ipynb)
as well as daily_labels.csv and monthly_labels.csv (see histograms_of_optimal_levels.py)

This script creates daily_compress_df and monthly_compress_df. (used by models.py)
"""

import pandas as pd
from os.path import exists
import re
import argparse
import numpy as np

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--labelloc", help="location of labels",
                        type=str, default=f"../../data/daily/daily_labels.csv")
    parser.add_argument("-c", "--calcloc", help="location of features",
                        type=str, default=f"../../data/daily/daily_calcs.csv")
    parser.add_argument("-o", "--output", help="location of output csv file",
                        type=str, default=f"../../data/daily/daily_df.csv")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parseArguments()
    argv_labelloc = args.labelloc
    argv_calcloc = args.calcloc
    argv_output = args.output

    daily_label_df = pd.read_csv(argv_labelloc)
    # monthly_label_df = pd.read_csv('../data/monthly_labels_zfp.csv')
    daily_calc_df = pd.read_csv(argv_calcloc)
    # monthly_calc_df = pd.read_csv('../data/monthly_calcs.csv')

    daily_df = pd.DataFrame()
    # lazy, for now just assuming the variables and timesteps are in identical order
    # No longer lazy, match variable name and timestep
    len(daily_calc_df)
    varnames = []
    for row in daily_calc_df.itertuples():
        print (row[1])
        varnames.append(re.search(r'orig_(?P<name>.*)', row[1])[1])
    daily_calc_df['variable'] = varnames

    daily_df = pd.merge(daily_calc_df, daily_label_df, on=['variable', 'time'])

    # monthly_df = pd.DataFrame()
    # monthly_df = monthly_calc_df
    # # lazy, for now just assuming the variables and timesteps are in identical order
    # monthly_df["levels"] = monthly_label_df["levels"]
    # monthly_df["algs"] = monthly_label_df["algs"]
    # monthly_df["ratios"] = monthly_label_df["ratios"]
    # monthly_df["variable"] = monthly_label_df["variable"]

    d_fileloc = argv_output
    # m_fileloc = f"../data/monthly_compress_df.csv"
    dfile_exists = exists(d_fileloc)
    # mfile_exists = exists(m_fileloc)
    if dfile_exists:
        daily_df.to_csv(d_fileloc, mode="a", header=False, index=False)
    else:
        daily_df.to_csv(d_fileloc, mode="a", header=True, index=False)

    # if mfile_exists:
    #     monthly_df.to_csv(m_fileloc, mode="a", header=False, index=False)
    # else:
    #     monthly_df.to_csv(m_fileloc, mode="a", header=True, index=False)
