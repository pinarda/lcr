import csv
import re
import os
import lcr_global_vars

def split_dssims(csvfilename, variable):
    """
    Split daily/monthly_dssims.csv into each variable.
    """

    location = f"../data/by_var/{variable}_calcs.csv"
    file_exists = os.path.isfile(location)
    with open(location, 'a', newline='') as csvfile:
        fieldnames = [
            'set',
            'time',
            'ssim_fp',
            'ks_p_value',
            'spatial_rel_error',
            'max_spatial_rel_error',
            'pearson_correlation_coefficient'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        with open(csvfilename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                m = re.search('^(?P<compression>.*?)_(?P<level>[0-9]+?)_(?P<varname>.*)$', row[0])
                if (m is not None):
                    if (m.group('varname') == variable):
                        writer.writerow({
                            'set' : row[0],
                            'time' : row[1],
                            'ssim_fp' : row[2],
                            'ks_p_value' : row[3],
                            'spatial_rel_error' : row[4],
                            'max_spatial_rel_error' : row[5],
                            'pearson_correlation_coefficient' : row[6]
                        })


if __name__ == "__main__":
    for v in lcr_global_vars.daily_vars:
        split_dssims("../data/daily_dssims.csv", v)