import csv
import numpy as np
import os

def optimal_per_slice():
    freqs = ["daily", "monthly"]
    for freq in freqs:
        csvfilename = f"../data/{freq}_zfp_bg_sz_comp_slices.csv"
        with open(csvfilename, newline='') as csvfile:
            content = csvfile.readlines()
        rowstrings = content[1:]
        for rowstring in rowstrings:
            row = rowstring.strip().split(",")
            variable = row[0]
            ratios = [row[5], row[8], row[11]]
            np_ratios = np.array(ratios)
            best_ratio = max(ratios)
            best_size = min(row[4], row[7], row[10])
            best_level = row[3 + np_ratios.argmax(0) * 3]
            if np_ratios.argmax(0) == 0:
                best_alg = "bg"
            elif np_ratios.argmax(0) == 1:
                best_alg = "zfp"
            elif np_ratios.argmax(0) == 2:
                best_alg = "sz"
            location = f"../data/{freq}_optimal_slices.csv"
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

def optimal_over_var():
    freqs = ["daily", "monthly"]
    for freq in freqs:
        csvfilename = f"../data/{freq}_zfp_bg_sz_comparison.csv"
        with open(csvfilename, newline='') as csvfile:
            content = csvfile.readlines()
        rowstrings = content[1:]
        for rowstring in rowstrings:
            row = rowstring.strip().split(",")
            variable = row[0]
            ratios = [row[3], row[6], row[9]]
            np_ratios = np.array(ratios)
            best_ratio = max(ratios)
            best_size = min(row[2], row[5], row[8])
            best_level = row[1 + np_ratios.argmax(0)*3]
            if np_ratios.argmax(0) == 0:
                best_alg = "bg"
            elif np_ratios.argmax(0) == 1:
                best_alg = "zfp"
            elif np_ratios.argmax(0) == 2:
                best_alg = "sz"
            location = f"../data/{freq}_optimal_over_var.csv"
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

if __name__ == "__main__":
    optimal_per_slice()