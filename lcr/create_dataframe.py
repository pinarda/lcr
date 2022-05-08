"""
This script requires daily_calcs.csv, monthly_calcs.csv (from DataGathering.ipynb)
as well as daily_labels.csv and monthly_labels.csv (see histograms_of_optimal_levels.py)

This script creates daily_compress_df and monthly_compress_df. (used by models.py)
"""

import pandas as pd
from os.path import exists

if __name__ == "__main__":
    daily_label_df = pd.read_csv('../data/bg_only_dssim/daily_labels.csv')
    # monthly_label_df = pd.read_csv('../data/monthly_labels.csv')
    daily_calc_df = pd.read_csv('../data/bg_only_dssim/daily_calcs.csv')
    # monthly_calc_df = pd.read_csv('../data/monthly_calcs.csv')

    daily_df = pd.DataFrame()
    daily_df = daily_calc_df
    # lazy, for now just assuming the variables and timesteps are in identical order
    daily_df["levels"] = daily_label_df["levels"]
    daily_df["algs"] = daily_label_df["algs"]
    daily_df["ratios"] = daily_label_df["ratios"]
    daily_df["variable"] = daily_label_df["variable"]

    # monthly_df = pd.DataFrame()
    # monthly_df = monthly_calc_df
    # # lazy, for now just assuming the variables and timesteps are in identical order
    # monthly_df["levels"] = monthly_label_df["levels"]
    # monthly_df["algs"] = monthly_label_df["algs"]
    # monthly_df["ratios"] = monthly_label_df["ratios"]
    # monthly_df["variable"] = monthly_label_df["variable"]

    d_fileloc = f"../data/daily_compress_df.csv"
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
