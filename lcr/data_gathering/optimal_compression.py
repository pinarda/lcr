"""
Contains the code necessary to extract a list of optimal compression values from a csv file containing
columns corresponding to {compression_type}_{level}, {variable}, {time}, and {DSSIM}

It would be best to open the csv file once, and get a list of all variables, levels, and timesteps
so I don't read the csv file more times than necessary. Seems like search_csv does most of what I need
already.

REQUIRES: daily/monthly_dssims.csv
"""

import csv
import re
import numpy as np
import os
import lcr_global_vars
import sys
import argparse


def search_csv(csvfilename: str, variable: str, timestep: int, compression:str):
    """
    Searches csv file for an entry with the given variable in the first column
    in the format .*_\d+_VARIABLE, and timestep given in the second column.
    """
    match_rows = []
    with open(csvfilename, newline='') as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
            if len(row) == 0:
                continue
            if compression != "sz3":
                m = re.search('(?P<compression>.*?)_(?P<level>[0-9]+?)_(?P<varname>.*)', row[0])
                time = row[1]
                if(m is not None):
                    if (m.group("varname") == variable and str(timestep) == time and m.group("compression") == compression):
                        match_rows.append(row)
            else:
                m = re.compile(r'(?P<level>[510][^_]*)_(?P<varname>.*)').findall(row[0])
                time = row[1]
                if(len(m) > 0):
                    if (m[0][1] == variable and str(timestep) == time):
                        match_rows.append(row)
    return match_rows


def optimal_level(csvfilename: str, variable: str, timestep: int, threshold: float, compression: str):
    """
    Finds the optimal compression level in a csv file assuming the levels are in the first
    column with the format .*_LEVEL_.* and the DSSIM/comparison values are in the third column.
    """
    rows = search_csv(csvfilename, variable, timestep, compression)
    if len(rows) == 0:
        return 0
    levels = []

    # ensure unique variable/level/timeslice
    rowids = []
    for row in rows:
        rowid = row[0] + row[1]
        rowids.append(rowid)
    rows = [rows[i] for i in np.unique(rowids, return_index=True)[1][::-1]]

    # ensure list of levels is in descending order (i.e. least compressed first)

    if compression not in ["sz1.4", "sz1ROn", "sz3"]:
        for row in rows:
            m = re.search('.*?_(?P<level>[0-9]+?)_(?P<varname>.*)', row[0])
            levels.append(int(m.group("level")))
            sort_index = np.argsort(levels)
        rows = [rows[i] for i in sort_index[::-1]]
        levels = [levels[i] for i in sort_index[::-1]]
    if compression in ["sz1.4", "sz1ROn", "sz3"]:
        for row in rows:
            m = re.search('.*?_(?P<level>[0-9]+?)_(?P<varname>.*)', row[0])
            levels.append(m.group("level"))
        rows = rows[::-1]
        levels = levels[::-1]

    # compute optimal level based on dssim
    i = 0
    prev_lev = None
    for row in rows:
        dssim = float(row[2])
        if dssim >= threshold:
            prev_lev=levels[i]
            i=i+1
            continue
        if dssim < threshold:
            if prev_lev is not None:
                best_lev = prev_lev
                return best_lev
            else:
                return -1
    return levels[len(levels)-1]

def optimal_level_multiple_comparison(csvfilename: str, variable: str, timestep: int,
                                      dssim_threshold: float, ks_p_threshold: float,
                                      spatial_err_threshold: float, max_spatial_err_threshold: float,
                                      pcc_threshold: float, compression: str):
    """
    Finds the optimal compression level in a csv file assuming the levels are in the first
    column with the format .*_LEVEL_.* the DSSIM/comparison values are in the third column, fourth, ... columns.
    """
    rows = search_csv(csvfilename, variable, timestep, compression)
    if len(rows) == 0:
        return 0
    levels = []

    # ensure unique variable/level/timeslice
    rowids = []
    for row in rows:
        rowid = row[0] + row[1]
        rowids.append(rowid)
    rows = [rows[i] for i in np.unique(rowids, return_index=True)[1][::-1]]

    # ensure list of levels is in descending order (i.e. least compressed first)

    if compression not in ["sz1.4", "sz1ROn", "sz3"]:
        for row in rows:
            m = re.search('.*?_(?P<level>[0-9]+?)_(?P<varname>.*)', row[0])
            levels.append(int(m.group("level")))
            sort_index = np.argsort(levels)
        rows = [rows[i] for i in sort_index[::-1]]
        levels = [levels[i] for i in sort_index[::-1]]
    if compression in ["sz1.4", "sz1ROn", "sz3"]:
        if compression == "sz3":
            for row in rows:
                m = re.compile(r'(?P<level>[510][^_]*)_(?P<varname>.*)').findall(row[0])
                level = m[0][0]
                levels.append(level)
            rows = rows[::-1]
            levels = levels[::-1]
        else:
            for row in rows:
                m = re.search('.*?_(?P<level>[0-9]+?)_(?P<varname>.*)', row[0])
                levels.append(m.group("level"))
            rows = rows[::-1]
            levels = levels[::-1]

    # compute optimal level based on dssim
    i = 0
    prev_lev = None
    best_dssim_lev = -1
    for row in rows:
        dssim = float(row[2])
        if dssim >= dssim_threshold:
            prev_lev=levels[i]
            i=i+1
            continue
        if dssim < dssim_threshold:
            if prev_lev is not None:
                best_dssim_lev = prev_lev
            else:
                best_dssim_lev = 100000

    if best_dssim_lev == -1:
        best_dssim_lev = prev_lev

    i = 0
    prev_lev = None
    best_ks_p_lev = -1
    for row in rows:
        ks_p = float(row[3])
        if ks_p >= ks_p_threshold:
            prev_lev=levels[i]
            i=i+1
            continue
        if ks_p < ks_p_threshold:
            if prev_lev is not None:
                best_ks_p_lev = prev_lev
            else:
                best_ks_p_lev = 100000


    if best_ks_p_lev == -1:
        best_ks_p_lev = prev_lev

    i = 0
    prev_lev = None
    best_spatial_err_lev = -1
    for row in rows:
        spatial_err = 100-float(row[4])
        if spatial_err >= spatial_err_threshold:
            prev_lev = levels[i]
            i = i + 1
            continue
        if spatial_err < spatial_err_threshold:
            if prev_lev is not None:
                best_spatial_err_lev = prev_lev
            else:
                best_spatial_err_lev = 100000


    if best_spatial_err_lev == -1:
        best_spatial_err_lev = prev_lev

    i = 0
    prev_lev = None
    best_max_spatial_err_lev = -1
    for row in rows:
        max_spatial_err = 1-float(row[5])
        if max_spatial_err >= max_spatial_err_threshold:
            prev_lev = levels[i]
            i = i + 1
            continue
        if max_spatial_err < max_spatial_err_threshold:
            if prev_lev is not None:
                best_max_spatial_err_lev = prev_lev
            else:
                best_max_spatial_err_lev = 100000

    if best_max_spatial_err_lev == -1:
        best_max_spatial_err_lev = prev_lev

    i = 0
    prev_lev = None
    best_pcc_lev = -1
    for row in rows:
        pcc = float(row[6])
        if pcc >= pcc_threshold:
            prev_lev = levels[i]
            i = i + 1
            continue
        if pcc < pcc_threshold:
            if prev_lev is not None:
                best_pcc_lev = prev_lev
            else:
                best_pcc_lev = 100000

    if best_pcc_lev == -1:
        best_pcc_lev = prev_lev

    levs = [float(best_dssim_lev), float(best_ks_p_lev), float(best_spatial_err_lev), float(best_max_spatial_err_lev), float(best_pcc_lev)]

    if compression == "sz3":
        return levs, min(levs)
    return levs, max(levs)


def optimal_level_max(csvfilename, variable, threshold, compression, freq, argv_var):
    """
    Find the minimum of all the optimal compression levels for a specified variable
    over all time slices.
    """
    times = []
    with open(csvfilename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if compression != "sz3":
                m = re.search('.*?_(?P<level>[0-9]+?)_(?P<varname>.*)', row[0])
                time = row[1]
                if (m is not None):
                    if (m.group("varname") == variable):
                        times.append(time)
            else:
                m = re.compile(r'(?P<level>[510][^_]*)_(?P<varname>.*)').findall(row[0])
                time = row[1]
                if (len(m) > 0):
                    if (m[0][1] == variable):
                        times.append(time)
    times = np.unique(times)

    levs = []
    for time in times:
        #index, lev = optimal_level_multiple_comparison(f"../../data/{freq}_dssims.csv", variable, time, threshold, 0.05, 100-5, 1-0.05, 0.99999, compression)
        lev = optimal_level(f"/glade/scratch/apinard/{freq}{argv_var}.csv", variable, time, threshold, compression)
        levs.append(lev)
    min_level = max(levs)
    return min_level

def optimal_level_spread(csvfilename, variable, threshold, compression, freq, argv_var):
    """
    Find the minimum of all the optimal compression levels for a specified variable
    over all time slices.
    """
    times = []
    with open(csvfilename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 0:
                continue
            if compression != "sz3":
                m = re.search('.*?_(?P<level>[0-9]+?)_(?P<varname>.*)', row[0])
                time = row[1]
                if(m is not None):
                    if (m.group("varname") == variable):
                        times.append(time)
            else:
                m = re.compile(r'(?P<level>[510][^_]*)_(?P<varname>.*)').findall(row[0])
                time = row[1]
                if (len(m) > 0):
                    if (m[0][1] == variable):
                        times.append(time)

    times = np.unique(times)

    levs = []
    all_levs = []
    for time in times:
        # print(variable)
        # print(time)
        # print(threshold)
        # print(compression)
        # print(freq)
        # print(argv_var)
        # print(optimal_level_multiple_comparison(f"/glade/scratch/apinard/monthly{argv_var}.csv", variable, time, threshold, 0.05, 100-5, 1-0.05, 0.99999, compression))
        all_lev, lev = optimal_level_multiple_comparison(f"/glade/scratch/apinard/{freq}{argv_var}.csv", variable, time, threshold, 0.05, 100-5, 1-0.05, 0.99999, compression)

        #lev = optimal_level(f"/glade/scratch/apinard/sz3/{argv_var}_calcs.csv", variable, time, threshold, compression)

        levs.append(lev)
        all_levs.append(all_lev)
    return all_levs, levs



def filesize(csvfilename, variable, level, compression):
    with open(csvfilename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        if compression == "sz3":
            for row in reader:
                if len(row) == 0:
                    return -1
                if level == "orig" or level == 100000:
                    if row[0] == variable and row[1] == f"orig":
                        return row[2]
                if row[0] == variable and row[1] == f"{compression}_ROn{level}":
                    return row[2]
        else:
            for row in reader:
                if len(row) == 0:
                    return -1
                if level == "orig" or int(level) == 100000:
                    if row[0] == variable and row[1] == f"orig":
                        return row[2]
                if row[0] == variable and row[1] == f"{compression}_{level}":
                    return row[2]

def create_daily_monthly_freq_hist():
    for freq in ['daily', 'monthly']:
        v = lcr_global_vars.varlist(f"../data/{freq}_dssims.csv")
        for varname in v:
            all_levs, level = optimal_level_spread(f"../data/{freq}_dssims.csv", varname, 0.995, "bg", freq)
            bg_levels=[2, 3, 4, 5, 6, 7]
            hist = {}
            for l in bg_levels:
                hist[l] = level.count(l)
            location = f"../data/test{freq}_bg_hist.csv"
            file_exists = os.path.isfile(location)
            with open(location, 'a', newline='') as csvfile:
                fieldnames = [
                    'variable',
                    'frequency',
                    '2',
                    '3',
                    '4',
                    '5',
                    '6',
                    '7'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()
                writer.writerow(
                    {
                        'variable': varname,
                        'frequency': freq,
                        '2': hist[2],
                        '3': hist[3],
                        '4': hist[4],
                        '5': hist[5],
                        '6': hist[6],
                        '7': hist[7]
                    }
                )


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--var", help="csv file to store output (if file exists, then data will append).",
                        type=str, default="./sample.csv")
    parser.add_argument("-f", "--freq", help="frequency (daily/monthly)",
                        type=str, default="monthly")
    args = parser.parse_args()

    return args


def main_zfp(argv):
    # Get command line stuff and store in a dictionary
    args = parseArguments()
    argv_var = args.var
    print(f"current_var: {argv_var}")

    for freq in ['daily']:
        # v = lcr_global_vars.varlist(f"../data/{freq}_dssims.csv")
        # for argv_var in v:
        location = f"../../data/{freq}/{freq}_zfp_bg_sz_comp_slices12345.csv"
        #location = f"../data/monthly_zfp_bg_sz_comp_slices.csv"
        file_exists = os.path.isfile(location)
        with open(location, 'a', newline='') as csvfile:
            fieldnames = [
                'variable',
                'frequency',
                'timestep',
                'br_level',
                'br_size',
                'br_ratio',
                'zfp_level',
                'zfp_size',
                'zfp_ratio',
                'sz_level',
                'sz_size',
                'sz_ratio',
                "all_br_levs",
                "all_zfp_levs",
                'all_sz_levs'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

        print(f"current_var: {argv_var}")
        # all_bg_levs, levelbg = optimal_level_spread(f"../data/daily_dssims.csv", argv_var, 0.9995, "br", freq, argv_var)

        all_bg_levs, levelbg = optimal_level_spread(f"/glade/scratch/apinard/monthly{argv_var}.csv", argv_var, 0.995, "br", freq, argv_var)
        print(f"level bg: {levelbg}")
        levelbg = [int(i) for i in levelbg]
        # levelzfp = optimal_level_spread(f"../../data/monthly_dssims.csv", argv_var, 0.9995, "zfp_p", freq, argv_var)
        # levelsz = optimal_level_spread(f"../../data/monthly_dssims.csv", argv_var, 0.9995, "sz3", freq, argv_var)

        all_zfp_levs, levelzfp = optimal_level_spread(f"/glade/scratch/apinard/monthly{argv_var}.csv", argv_var, 0.995, "zfp_p", freq, argv_var)
        all_sz_levs, levelsz = optimal_level_spread(f"/glade/scratch/apinard/monthly{argv_var}.csv", argv_var, 0.995, "sz3", freq, argv_var)
        print(levelsz)
        levelzfp = [int(i) for i in levelzfp]

        levelsz = [str(i) for i in levelsz]

        location = f"../../data/{freq}/{freq}_zfp_bg_sz_comp_slices12345.csv"
        #location = f"../data/monthly_zfp_bg_sz_comp_slices.csv"
        file_exists = os.path.isfile(location)
        with open(location, 'a', newline='') as csvfile:
            fieldnames = [
                'variable',
                'frequency',
                'timestep',
                'br_level',
                'br_size',
                'br_ratio',
                'zfp_level',
                'zfp_size',
                'zfp_ratio',
                'sz_level',
                'sz_size',
                'sz_ratio',
                "all_br_levs",
                "all_zfp_levs",
                'all_sz_levs'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            sizecsv = f"../../data/{freq}/{freq}_filesizes.csv"

            for i in range(0, 360):
                print(f"{i}")
                print(sizecsv)
                print(argv_var)
                print(levelzfp[i])
                fzfp = filesize(sizecsv, argv_var, levelzfp[i], "zfp_p")
                print(fzfp)
                fbg = filesize(sizecsv, argv_var, levelbg[i], "br")
                print(sizecsv)
                print(argv_var)
                print(levelsz[i])
                fsz = filesize(sizecsv, argv_var, levelsz[i], "sz3")
                print(fsz)
                if fsz is not None or fbg is not None or fzfp is not None:
                    sizesz = float(fsz)
                    sizebg = float(fbg)
                    sizezfp = float(fzfp)
                    ratiosz = float(filesize(sizecsv, argv_var, "orig", "sz3")) / float(fsz)
                    ratiobg = float(filesize(sizecsv, argv_var, "orig", "br")) / float(fbg)
                    ratiozfp = float(filesize(sizecsv, argv_var, "orig", "zfp")) / float(fzfp)
                writer.writerow(
                    {
                        'variable': argv_var,
                        'frequency': freq,
                        'timestep': i,
                        'br_level': levelbg[i],
                        'br_size': sizebg,
                        'br_ratio': ratiobg,
                        "all_br_levs": all_bg_levs[i],
                        'zfp_level': levelzfp[i],
                        'zfp_size': sizezfp,
                        'zfp_ratio': ratiozfp,
                        "all_zfp_levs": all_zfp_levs[i],
                        'sz_level': levelsz[i],
                        'sz_size': sizesz,
                        'sz_ratio': ratiosz,
                        'all_sz_levs': all_sz_levs[i]
                    }
                )


if __name__ == "__main__":
    main_zfp(sys.argv[1:])

# if __name__ == "__main__":
#     args = parseArguments()
#     argv_var = args.var
#     #daily_sizecsv = "../data/daily_filesizes.csv"
#    # varname = "TS"
#     # sz_level = optimal_level_max(f"../data/daily_dssims.csv", "TS", 0.9995, "sz1.4", "daily")
#     # f = filesize(daily_sizecsv, varname, sz_level, "sz1.4")
#     monthly_sizecsv = "../../data/daily_filesizes.csv"
#     daily_sizecsv = "../../data/daily_filesizes.csv"
#     for num in [0.95, 0.995, 0.9995]:
#         for freq in ['daily']:
#             v = lcr_global_vars.varlist(f"../../data/{freq}_dssims.csv")
#             v = [argv_var]
#             for varname in v:
#                 level = optimal_level_max(f"../../data/by_var/{argv_var}_calcs.csv", varname, num, "bg", freq, varname)
#                 #level = optimal_level_max(f"../../data/{freq}_dssims.csv", varname, num, "bg", freq, varname)
#
#                 f = filesize(monthly_sizecsv, varname, level, "bg")
#                 if f is not None:
#                     size = float(f)
#                     ratio = float(filesize(monthly_sizecsv, varname, "orig", "bg"))/float(f)
#                 else:
#                     size = float(filesize(monthly_sizecsv, varname, level, "bg"))
#                     ratio = float(filesize(monthly_sizecsv, varname, "orig", "bg")) / float(filesize(monthly_sizecsv, varname, level, "bg"))
#
#                 # zfp_level = optimal_level_max(f"../../data/by_var/{argv_var}_calcs.csv", varname, num, "zfp_p", freq, varname)
#                 # if freq == "daily":
#                 #     f = filesize(daily_sizecsv, varname, zfp_level, "zfp")
#                 # elif freq == "monthly":
#                 #     f = filesize(monthly_sizecsv, varname, zfp_level, "zfp")
#                 # if f is not None:
#                 #     zfp_size = float(f)
#                 #     zfp_ratio = float(filesize(monthly_sizecsv, varname, "orig", "zfp")) / float(f)
#                 # else:
#                 #     zfp_size = float(filesize(monthly_sizecsv, varname, zfp_level, "zfp"))
#                 #     zfp_ratio = float(filesize(monthly_sizecsv, varname, "orig", "zfp")) / float(
#                 #         filesize(monthly_sizecsv, varname, zfp_level, "zfp"))
#
#                 # sz_level = optimal_level_max(f"../data/test_set/{freq}_dssims.csv", varname, 0.9995, "sz1.4", freq)
#                 # f = filesize(daily_sizecsv, varname, sz_level, "sz1.4")
#                 # if f is not None:
#                 #     sz_size = float(f)
#                 #     sz_ratio = float(filesize(daily_sizecsv, varname, "orig", "sz1.4")) / float(f)
#                 # else:
#                 #     sz_size = float(filesize(monthly_sizecsv, varname, sz_level, "sz1.4"))
#                 #     sz_ratio = float(filesize(monthly_sizecsv, varname, "orig", "sz1.4")) / float(
#                 #         filesize(monthly_sizecsv, varname, sz_level, "sz1.4"))
#
#                 location = f"../../data/{freq}_zfp_bg_sz_comparison_test_{num}.csv"
#                 file_exists = os.path.isfile(location)
#                 with open(location, 'a', newline='') as csvfile:
#                     fieldnames = [
#                         'variable',
#                         'bg_level',
#                         'bg_size',
#                         'bg_ratio',
#                         # 'zfp_level',
#                         # 'zfp_size',
#                         # 'zfp_ratio',
#                         #'sz_level',
#                         #'sz_size',
#                         #'sz_ratio'
#                     ]
#                     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#
#                     if not file_exists:
#                         writer.writeheader()
#                     writer.writerow(
#                         {
#                             'variable': varname,
#                             'bg_level': level,
#                             'bg_size': size,
#                             'bg_ratio' : ratio,
#                             # 'zfp_level': zfp_level,
#                             # 'zfp_size': zfp_size,
#                             # 'zfp_ratio': zfp_ratio,
#                             #'sz_level': sz_level,
#                             #'sz_size': sz_size,
#                             #'sz_ratio': sz_ratio
#                         }
#                     )



# if __name__ == "__main__":
#
#     for freq in ['daily', 'monthly']:
#         v = lcr_global_vars.varlist(f"../data/{freq}_dssims.csv")
#         for varname in v:
#             level = optimal_level_spread(f"../data/{freq}_dssims.csv", varname, 0.9995, "sz1.4", freq)
#             location = f"../data/{freq}_sz14_optimal_slices.csv"
#             file_exists = os.path.isfile(location)
#             with open(location, 'a', newline='') as csvfile:
#                 fieldnames = [
#                     'variable',
#                     'frequency',
#                     '0',
#                     '1',
#                     '2',
#                     '3',
#                     '4',
#                     '5',
#                     '6',
#                     '7',
#                     '8',
#                     '9',
#                     '10',
#                     '11',
#                     '12',
#                     '13',
#                     '14',
#                     '15',
#                     '16',
#                     '17',
#                     '18',
#                     '19',
#                     '20',
#                     '21',
#                     '22',
#                     '23',
#                     '24',
#                     '25',
#                     '26',
#                     '27',
#                     '28',
#                     '29',
#                     '30',
#                     '31',
#                     '32',
#                     '33',
#                     '34',
#                     '35',
#                     '36',
#                     '37',
#                     '38',
#                     '39',
#                     '40',
#                     '41',
#                     '42',
#                     '43',
#                     '44',
#                     '45',
#                     '46',
#                     '47',
#                     '48',
#                     '49',
#                     '50',
#                     '51',
#                     '52',
#                     '53',
#                     '54',
#                     '55',
#                     '56',
#                     '57',
#                     '58',
#                     '59'
#                 ]
#                 writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#
#                 if not file_exists:
#                     writer.writeheader()
#                 writer.writerow(
#                     {
#                         'variable': varname,
#                         'frequency': freq,
#                         '0': level[0],
#                         '1': level[1],
#                         '2': level[2],
#                         '3': level[3],
#                         '4': level[4],
#                         '5': level[5],
#                         '6': level[6],
#                         '7': level[7],
#                         '8': level[8],
#                         '9': level[9],
#                         '10': level[10],
#                         '11': level[11],
#                         '12': level[12],
#                         '13': level[13],
#                         '14': level[14],
#                         '15': level[15],
#                         '16': level[16],
#                         '17': level[17],
#                         '18': level[18],
#                         '19': level[19],
#                         '20': level[20],
#                         '21': level[21],
#                         '22': level[22],
#                         '23': level[23],
#                         '24': level[24],
#                         '25': level[25],
#                         '26': level[26],
#                         '27': level[27],
#                         '28': level[28],
#                         '29': level[29],
#                         '30': level[30],
#                         '31': level[31],
#                         '32': level[32],
#                         '33': level[33],
#                         '34': level[34],
#                         '35': level[35],
#                         '36': level[36],
#                         '37': level[37],
#                         '38': level[38],
#                         '39': level[39],
#                         '40': level[40],
#                         '41': level[41],
#                         '42': level[42],
#                         '43': level[43],
#                         '44': level[44],
#                         '45': level[45],
#                         '46': level[46],
#                         '47': level[47],
#                         '48': level[48],
#                         '49': level[49],
#                         '50': level[50],
#                         '51': level[51],
#                         '52': level[52],
#                         '53': level[53],
#                         '54': level[54],
#                         '55': level[55],
#                         '56': level[56],
#                         '57': level[57],
#                         '58': level[58],
#                         '59': level[59],
#                     }
#                 )

    # for freq in ['daily', 'monthly']:
    #     v = lcr_global_vars.varlist(f"../data/{freq}_dssims.csv")
    #     for varname in v:
    #         level = optimal_level_spread(f"../data/{freq}_dssims.csv", varname, 0.9995, "zfp_p", freq)
    #         bg_levels=[8, 10, 12, 14, 16, 18, 20, 22, 24]
    #         hist = {}
    #         for l in bg_levels:
    #             hist[l] = level.count(l)
    #         location = f"../data/{freq}_zfp_hist.csv"
    #         file_exists = os.path.isfile(location)
    #         with open(location, 'a', newline='') as csvfile:
    #             fieldnames = [
    #                 'variable',
    #                 'frequency',
    #                 '8',
    #                 '10',
    #                 '12',
    #                 '14',
    #                 '16',
    #                 '18',
    #                 '20',
    #                 '22',
    #                 '24'
    #             ]
    #             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #
    #             if not file_exists:
    #                 writer.writeheader()
    #             writer.writerow(
    #                 {
    #                     'variable': varname,
    #                     'frequency': freq,
    #                     '8': hist[8],
    #                     '10': hist[10],
    #                     '12': hist[12],
    #                     '14': hist[14],
    #                     '16': hist[16],
    #                     '18': hist[18],
    #                     '20': hist[20],
    #                     '22': hist[22],
    #                     '24': hist[24]
    #                 }
    #             )
    #
    # for freq in ['daily', 'monthly']:
    #     v = lcr_global_vars.varlist(f"../data/{freq}_dssims.csv")
    #     for varname in v:
    #         level = optimal_level_spread(f"../data/{freq}_dssims.csv", varname, 0.9995, "sz1.4", freq)
    #         bg_levels=["1", "05", "01", "005", "001", "0005", "0001", "00005", "00001", "000005", "000001"]
    #         hist = {}
    #         for l in bg_levels:
    #             hist[l] = level.count(l)
    #         location = f"../data/{freq}_sz14_hist.csv"
    #         file_exists = os.path.isfile(location)
    #         with open(location, 'a', newline='') as csvfile:
    #             fieldnames = [
    #                 'variable',
    #                 'freq',
    #                 '1',
    #                 '05',
    #                 '01',
    #                 '005',
    #                 '001',
    #                 '0005',
    #                 '0001',
    #                 '00005',
    #                 '00001',
    #                 '000005',
    #                 '000001'
    #             ]
    #             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #
    #             if not file_exists:
    #                 writer.writeheader()
    #             writer.writerow(
    #                 {
    #                     'variable': varname,
    #                     'freq': freq,
    #                     '1': hist["1"],
    #                     '05': hist["05"],
    #                     '01': hist["01"],
    #                     '005': hist["005"],
    #                     '001': hist["001"],
    #                     '0005': hist["0005"],
    #                     '0001': hist["0001"],
    #                     '00005': hist["00005"],
    #                     '00001': hist["00001"],
    #                     '000005': hist["000005"],
    #                     '000001': hist["000001"]
    #                 }
    #             )




    # for freq in ['monthly', 'daily']:
    #     v = lcr_global_vars.varlist(f"../data/{freq}_dssims.csv")
    #
    #     location = f"../data/{freq}_zfp_bg_sz_comp_slices.csv"
    #     file_exists = os.path.isfile(location)
    #     with open(location, 'a', newline='') as csvfile:
    #         fieldnames = [
    #             'variable',
    #             'frequency',
    #             'timestep',
    #             'bg_level',
    #             'bg_size',
    #             'bg_ratio',
    #             'zfp_level',
    #             'zfp_size',
    #             'zfp_ratio',
    #             'sz_level',
    #             'sz_size',
    #             'sz_ratio',
    #             'sz1413_level',
    #             'sz1413_size',
    #             'zfp5_level',
    #             'zfp5_size',
    #             'zfp5_ratio'
    #         ]
    #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #         if not file_exists:
    #             writer.writeheader()
    #
    #     for varname in v:
    #         levelsz = optimal_level_spread(f"../data/{freq}_dssims.csv", varname, 0.9995, "sz1.4", freq)
    #         levelsz1413 = optimal_level_spread(f"../data/{freq}_dssims.csv", varname, 0.9995, "sz1ROn", freq)
    #         levelbg = optimal_level_spread(f"../data/{freq}_dssims.csv", varname, 0.9995, "bg", freq)
    #         levelzfp = optimal_level_spread(f"../data/{freq}_dssims.csv", varname, 0.9995, "zfp_p", freq)
    #         levelzfp5 = optimal_level_spread(f"../data/{freq}_dssims.csv", varname, 0.9995, "zfp5_p", freq)
    #         location = f"../data/{freq}_zfp_bg_sz_comp_slices.csv"
    #         file_exists = os.path.isfile(location)
    #         with open(location, 'a', newline='') as csvfile:
    #             fieldnames = [
    #                 'variable',
    #                 'frequency',
    #                 'timestep',
    #                 'bg_level',
    #                 'bg_size',
    #                 'bg_ratio',
    #                 'zfp_level',
    #                 'zfp_size',
    #                 'zfp_ratio',
    #                 'sz_level',
    #                 'sz_size',
    #                 'sz_ratio',
    #                 'sz1413_level',
    #                 'sz1413_size',
    #                 'sz1413_ratio',
    #                 'zfp5_level',
    #                 'zfp5_size',
    #                 'zfp5_ratio'
    #             ]
    #             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #             sizecsv = f"../data/{freq}_filesizes.csv"
    #
    #             for i in range(0, 60):
    #                 fsz = filesize(sizecsv, varname, levelsz[i], "sz1.4")
    #                 fsz1413 = filesize(sizecsv, varname, levelsz1413[i], "sz1ROn")
    #                 fzfp = filesize(sizecsv, varname, levelzfp[i], "zfp_p")
    #                 fbg = filesize(sizecsv, varname, levelbg[i], "bg")
    #                 fzfp5 = filesize(sizecsv, varname, levelzfp5[i], "zfp5")
    #                 if fsz is not None:
    #                     sizesz = float(fsz)
    #                     sizesz1413 = float(fsz1413)
    #                     sizezfp = float(fzfp)
    #                     sizebg = float(fbg)
    #                     sizezfp5 = float(fzfp5)
    #                     ratiosz = float(filesize(sizecsv, varname, "orig", "sz1.4")) / float(fsz)
    #                     ratiosz1413 = float(filesize(sizecsv, varname, "orig", "sz1ROn")) / float(fsz1413)
    #                     ratiozfp = float(filesize(sizecsv, varname, "orig", "zfp_p")) / float(fzfp)
    #                     ratiobg = float(filesize(sizecsv, varname, "orig", "bg")) / float(fbg)
    #                     ratiozfp5 = float(filesize(sizecsv, varname, "orig", "zfp5")) / float(fzfp5)
    #                 writer.writerow(
    #                     {
    #                         'variable': varname,
    #                         'frequency': freq,
    #                         'timestep': i,
    #                         'bg_level': levelbg[i],
    #                         'bg_size': sizebg,
    #                         'bg_ratio': ratiobg,
    #                         'zfp_level': levelzfp[i],
    #                         'zfp_size': sizezfp,
    #                         'zfp_ratio': ratiozfp,
    #                         'sz_level': levelsz[i],
    #                         'sz_size': sizesz,
    #                         'sz_ratio': ratiosz,
    #                         'sz1413_level': levelsz1413[i],
    #                         'sz1413_size': sizesz1413,
    #                         'sz1413_ratio': ratiosz1413,
    #                         'zfp5_level': levelzfp5[i],
    #                         'zfp5_size': sizezfp5,
    #                         'zfp5_ratio': ratiozfp5,
    #                     }
    #                 )