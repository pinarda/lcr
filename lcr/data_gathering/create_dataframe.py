"""
This script requires daily_calcs.csv, monthly_calcs.csv (from DataGathering.ipynb)
as well as daily_labels.csv and monthly_labels.csv (see histograms_of_optimal_levels.py)

This script creates daily_compress_df and monthly_compress_df. (used by models.py)
"""

import pandas as pd
from os.path import exists
import numpy as np

if __name__ == "__main__":
    freqs = ["daily", "monthly"]
    for freq in freqs:
        daily_label_df = pd.read_csv(f'../../data/{freq}/{freq}_labels_zfp.csv')
        # monthly_label_df = pd.read_csv('../data/monthly_labels_zfp.csv')
        daily_calc_df = pd.read_csv(f'../../data/{freq}/{freq}_calcs.csv')
        # monthly_calc_df = pd.read_csv('../data/monthly_calcs.csv')

        daily_df2 = daily_calc_df.loc[daily_calc_df['time'] < 360].copy()
        daily_df = daily_df2.reset_index()
        # lazy, for now just assuming the variables and timesteps are in identical order
        daily_df["levels"] = daily_label_df["levels"].copy()
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

        d_fileloc = f"../../data/{freq}/{freq}_compress_df_zfp.csv"
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
