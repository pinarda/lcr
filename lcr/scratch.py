import math
import numpy as np
# Add ldcpy root to system path
import struct
import sys
from math import log2
from mpmath import log
import mpmath

import astropy
import matplotlib.pyplot as plt
import numpy as np

mpmath.dps = 50

sys.path.insert(0, '../../../')

# suppress all of the divide by zero warnings
import warnings

warnings.filterwarnings("ignore")

import ldcpy

def binary(num):
    return ''.join(f'{c:0>8b}' for c in struct.pack('!f', num))


# NOTE: They only look backward
def get_prev_bit(bit_pos):
    return [bit_pos[0] - 1, bit_pos[1]]


N_BITS = 32


def getbpe(data_array, x_index, title):
    dict_list_H = []
    for i in range(N_BITS - 1):
        new_dict = {"00": 0,
                    "01": 0,
                    "10": 0,
                    "11": 0}
        dict_list_H.append(new_dict)

    num_measurements = 0
    for y in range(1, data_array.shape[1]):
        for z in range(data_array.shape[2]):
            num_measurements += 1

            bit_pos = [y, z]
            current_data = data_array[x_index][y][z]
            current_data = binary(current_data)

            adj_data_index = get_prev_bit(bit_pos)
            y_adj, z_adj = adj_data_index
            adj_data = data_array[x_index][y_adj][z_adj]
            adj_data = binary(adj_data)

            for i in range(N_BITS - 1):
                current_bit = int(current_data[i])
                adjacent_bit = int(adj_data[i])

                p00 = p01 = p10 = p11 = 0
                if current_bit == 0 and adjacent_bit == 0:
                    p00 = 1
                elif current_bit == 0 and adjacent_bit == 1:
                    p01 = 1
                elif current_bit == 1 and adjacent_bit == 0:
                    p10 = 1
                elif current_bit == 1 and adjacent_bit == 1:
                    p11 = 1

                dict_list_H[i]["00"] += p00
                dict_list_H[i]["01"] += p01
                dict_list_H[i]["10"] += p10
                dict_list_H[i]["11"] += p11

    bit_pos_H = []
    Hs = []
    diff = []
    k=0
    for bit_pos_dict in dict_list_H:

        p00 = np.float128(bit_pos_dict["00"]) / num_measurements
        p01 = np.float128(bit_pos_dict["01"]) / num_measurements
        p10 = np.float128(bit_pos_dict["10"]) / num_measurements
        p11 = np.float128(bit_pos_dict["11"]) / num_measurements

        p0 = p00 + p01
        p1 = p10 + p11

        H = 0
        if p0 != 0:
            H -= p0 * log(p0, 2)
        if p1 != 0:
            H -= p1 * log(p1, 2)

        Hs.append(H)

        H0 = 0
        if p00 != 0:
            H0 += p00 * log(p00, 2)

        if p01 != 0:
            H0 += p01 * log(p01, 2)

        H1 = 0
        if p10 != 0:
            H1 += p10 * log(p10, 2)

        if p11 != 0:
            H1 += p11 * log(p11, 2)

        prob_H = -p0 * np.float128(H0) - p1 * np.float128(H1)

        bit_pos_H.append(prob_H)

        diff.append(H - prob_H)

        k=k+1

    compression_levels = [24, 22, 20, 18, 16, 14, 12, 10, 8]
    compression_levels = ["ZFP_" + str(x) for x in compression_levels]
    compression_levels = ["Orig"] + compression_levels

    # plt.plot(bit_pos_H)
    # plt.plot(Hs)
    plt.plot(diff)
    plt.ylabel("Information content")
    plt.xlabel("Bit position")
    # plt.title(title + " " + compression_levels[x_index] + " " + str(sum(diff)))
    # plt.legend(["H", "Conditional H", "I(b)"])

    # plt.show()

if __name__ == "__main__":
    # col_ts is a collection containing TS data
    col_ts = ldcpy.open_datasets(
        "cam-fv",
        ["TS"],
        [
            "/Users/alex/git/ldcpy/data/cam-fv/orig.TS.100days.nc",
            "/Users/alex/git/ldcpy/data/cam-fv/zfp1e-1.TS.100days.nc",
            "/Users/alex/git/ldcpy/data/cam-fv/zfp1.0.TS.100days.nc",
        ],
        ["orig", "zfpA1e-1", "zfpA1.0"],
    )
    # col_prect contains PRECT data
    col_prect = ldcpy.open_datasets(
        "cam-fv",
        ["PRECT"],
        [
            "/Users/alex/git/ldcpy/data/cam-fv/orig.PRECT.60days.nc",
            "/Users/alex/git/ldcpy/data/cam-fv/zfp1e-11.PRECT.60days.nc",
            "/Users/alex/git/ldcpy/data/cam-fv/zfp1e-7.PRECT.60days.nc",
        ],
        ["orig", "zfpA1e-11", "zfpA1e-7"],
    )

    ts_array = np.array(col_ts["TS"].isel(time=0).values)
    prect_array = np.array(col_prect["PRECT"].isel(time=0).values)

    compression_levels = [24, 22, 20, 18, 16, 14, 12, 10, 8]
    compression_levels = ["ZFP_" + str(x) for x in compression_levels]
    compression_levels = ["Orig"] + compression_levels

    for daily_variable in ["TS"]:
        arr = ts_array
        for i in range(arr.shape[0]):
            getbpe(arr, i, daily_variable)
        plt.title(daily_variable)
        plt.legend(compression_levels)
        plt.show()
        # print()
        # print()
        # print()
        # print()

