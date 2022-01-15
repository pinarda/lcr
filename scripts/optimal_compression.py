"""
It would be best to open the csv file once, and get a list of all variables, levels, and timesteps
so I don't read the csv file more times than necessary. Seems like search_csv does most of what I need
already.
"""

import csv
import re
import numpy as np
import sys
import os


def search_csv(csvfilename: str, variable: str, timestep: int, compression:str):
    """
    Searches csv file for an entry with the given variable in the first column
    in the format .*_\d+_VARIABLE, and timestep given in the second column.
    """
    match_rows = []
    with open(csvfilename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            m = re.search('(?P<compression>.*)_(?P<level>\d+)_(?P<varname>.*)', row[0])
            time = row[1]
            if(m is not None):
                if (m.group("varname") == variable and str(timestep) == time and m.group("compression") == compression):
                    match_rows.append(row)
    return match_rows


def optimal_level(csvfilename: str, variable: str, timestep: int, threshold: float, compression: str):
    """
    Finds the optimal compression level in a csv file assuming the levels are in the first
    column with the format .*_LEVEL_.* and the DSSIM/comparison values are in the third column.
    """
    rows = search_csv(csvfilename, variable, timestep, compression)
    levels = []

    # ensure unique variable/level/timeslice
    rowids = []
    for row in rows:
        rowid = row[0] + row[1]
        rowids.append(rowid)
    rows = [rows[i] for i in np.unique(rowids, return_index=True)[1][::-1]]

    # ensure list of levels is in descending order (i.e. least compressed first)
    for row in rows:
        m = re.search('.*_(?P<level>\d+)_(?P<varname>.*)', row[0])
        levels.append(int(m.group("level")))
        sort_index = np.argsort(levels)
    rows = [rows[i] for i in sort_index[::-1]]
    levels = [levels[i] for i in sort_index[::-1]]

    # compute optimal level based on dssim
    i = 0
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


def optimal_level_min(csvfilename, variable, threshold, compression):
    """
    Find the minimum of all the optimal compression levels for a specified variable
    over all time slices.
    """
    times = []
    with open(csvfilename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            m = re.search('.*_(?P<level>\d+)_(?P<varname>.*)', row[0])
            time = row[1]
            if(m is not None):
                if (m.group("varname") == variable):
                    times.append(time)
    times = np.unique(times)

    levs = []
    for time in times:
        lev = optimal_level("../data/dssims.csv", variable, time, threshold, compression)
        levs.append(lev)
    min_level = min(levs)
    return min_level

def varlist(csvfilename):
    """
    Gets list of variables in a csv file
    """
    vars = []
    with open(csvfilename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            m = re.search('.*_(?P<level>\d+)_(?P<varname>.*)', row[0])
            if m is not None:
                vars.append(m.group("varname"))
    vars = np.unique(vars)
    return vars

def filesize(csvfilename, variable, level, compression):
    with open(csvfilename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if level == "orig":
                if row[0] == variable and row[1] == f"orig":
                    return row[2]
            if row[0] == variable and row[1] == f"bg_{level}":
                return row[2]

if __name__ == "__main__":
    monthly_sizecsv = "../data/monthly_filesizes.csv"
    daily_sizecsv = "../data/daily_filesizes.csv"
    v = varlist("../data/dssims.csv")
    print(v)
    for varname in v:
        level = optimal_level_min("../data/dssims.csv", varname, 0.9995, "bg")
        f = filesize(daily_sizecsv, varname, level, "bg")
        if f is not None:
            size = float(f)
            ratio = float(filesize(daily_sizecsv, varname, "orig", "bg"))/float(f)
        else:
            size = float(filesize(monthly_sizecsv, varname, level, "bg"))
            ratio = float(filesize(monthly_sizecsv, varname, "orig", "bg")) / float(filesize(monthly_sizecsv, varname, level, "bg"))

        zfp_level = "None"
        zfp_size = 0
        zfp_ratio = 0

        location = "../data/zfp_bg_comparison.csv"
        file_exists = os.path.isfile(location)
        with open(location, 'a', newline='') as csvfile:
            fieldnames = [
                'variable',
                'bg_level',
                'bg_size',
                'bg_ratio',
                'zfp_level',
                'zfp_size',
                'zfp_ratio'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()
            writer.writerow(
                {
                    'variable': varname,
                    'bg_level': level,
                    'bg_size': size,
                    'bg_ratio' : ratio,
                    'zfp_level': zfp_level,
                    'zfp_size': zfp_size,
                    'zfp_ratio': zfp_ratio
                }
            )