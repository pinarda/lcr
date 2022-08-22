"""
requires: daily and monthly_zfp_bg_sz_comp_slices.csv (see optimal_compresssion.py)
creates: daily and monthly_optimal_slices.csv (for histograms_of_optimal_levels.csv)
"""

import csv
import numpy as np
import os
import argparse
import pandas as pd

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--var", help="csv file to store output (if file exists, then data will append).",
                        type=str, default="./sample.csv")
    parser.add_argument("-l", "--loc", help="location of optimal csv file",
                        type=str, default=f"../../data/daily/daily_optim.csv")
    parser.add_argument("-o", "--output", help="location of output csv file",
                        type=str, default=f"../../data/daily/daily_optim_algs.csv")
    parser.add_argument("-a", "--algs", help="compression algorithms",
                        type=str, nargs='+', required=True)
    args = parser.parse_args()

    return args

def optimal_per_slice():
    # Get command line stuff and store in a dictionary
    args = parseArguments()
    # argv_var = args.var
    argv_loc = args.loc
    argv_algs = args.algs
    argv_output = args.output

    comp_csv = pd.read_csv(argv_loc)
    for i in range(len(comp_csv)):
    # csvfilename = argv_loc
    # with open(csvfilename, newline='') as csvfile:
    #     content = csvfile.readlines()
    # rowstrings = content[1:]
    # for rowstring in rowstrings:
    #     row = rowstring.strip().split(",")

        variable = comp_csv["variable"][i]
        rationames = [f"{alg}_ratio" for alg in argv_algs]
        ratios = [comp_csv[x][i] for x in rationames]
        #ratios = [row[5], row[8], row[11]] #, row[14], row[17]]
        # only look at zfp
        #ratios = [0, float(row[8]), 0]
        np_ratios = np.array(ratios)

        best_ratio = max(ratios)

        best_alg = rationames[np.where(ratios == best_ratio)[0][0]].split("_")[0]
        best_size = comp_csv[f"{best_alg}_size"][i]
        #best_size = min(row[4], row[7], row[10])#, row[13], row[16])
        # only look at zfp
        #best_size = min(9999999999999999, float(row[7]), 9999999999999999)
        #best_level = row[3 + np_ratios.argmax(0) * 3]
        best_level = comp_csv[f"{best_alg}_level"][i]
        # if np_ratios.argmax(0) == 0:
        #     best_alg = "bg"
        # elif np_ratios.argmax(0) == 1:
        #     best_alg = "zfp"
        # elif np_ratios.argmax(0) == 2:
        #     best_alg = "sz"
        # elif np_ratios.argmax(0) == 3:
        #     best_alg = "sz1413"
        # elif np_ratios.argmax(0) == 4:
        #     best_alg = "z_hdf5"
        location = argv_output
        file_exists = os.path.isfile(location)
        with open(location, 'a', newline='') as newcsvfile:
            fieldnames = ["variable", "best_alg", "best_ratio", "best_size", "best_level"]
            writer = csv.DictWriter(newcsvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()
            writer.writerow(
                {
                    'variable': variable,
                    'best_alg': best_alg,
                    'best_ratio': best_ratio,
                    'best_size': best_size,
                    'best_level': best_level
                }
            )

# def optimal_over_var():
#     freqs = ["daily"]
#     for freq in freqs:
#         csvfilename = f"../../data/{freq}/{freq}_zfp_bg_sz_comp_slices.csv"
#         with open(csvfilename, newline='') as csvfile:
#             content = csvfile.readlines()
#         rowstrings = content[1:]
#         for rowstring in rowstrings:
#             row = rowstring.strip().split(",")
#             variable = row[0]
#             ratios = [row[3], row[6], row[9]]
#             #, row[12], row[15]]
#             np_ratios = np.array(ratios)
#             best_ratio = max(ratios)
#             best_size = min(row[2], row[5], row[8])#, row[11], row[14])
#             best_level = row[1 + np_ratios.argmax(0)*3]
#             if np_ratios.argmax(0) == 0:
#                 best_alg = "bg"
#             elif np_ratios.argmax(0) == 1:
#                 best_alg = "zfp"
#             elif np_ratios.argmax(0) == 2:
#                 best_alg = "sz"
#             elif np_ratios.argmax(0) == 3:
#                 best_alg = "sz1413"
#             elif np_ratios.argmax(0) == 4:
#                 best_alg = "z_hdf5"
#             location = f"../../data/{freq}/{freq}_optimal_over_var.csv"
#             file_exists = os.path.isfile(location)
#             with open(location, 'a', newline='') as newcsvfile:
#                 fieldnames = ["variable", "best_alg", "best_ratio", "best_size", "best_level"]
#                 writer = csv.DictWriter(newcsvfile, fieldnames=fieldnames)
#
#                 if not file_exists:
#                     writer.writeheader()
#                 writer.writerow(
#                     {
#                      'variable': variable,
#                      'best_alg': best_alg,
#                      'best_ratio': best_ratio,
#                      'best_size': best_size,
#                      'best_level': best_level
#                     }
#                 )

if __name__ == "__main__":
    optimal_per_slice()