import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import ast
mpl.use( 'tkagg' )

if __name__ == "__main__":
    csvfilename = "../data/daily_zfp_bg_sz_comp_slices.csv"
    dssim_csvfilename = "../data/daily_zfp_bg_sz_comp_slices_dssim.csv"
    with open(csvfilename, newline='') as csvfile:
        with open(dssim_csvfilename, newline='') as dssim_csvfile:
            reader = csv.reader(csvfile)
            dssim_reader = csv.reader(dssim_csvfile)
            reader.__next__()
            dssim_reader.__next__()
            bg_diffs = []
            zfp_diffs = []
            v = []
            time = []
            for row in reader:
                # print(row)
                dssim_row = dssim_reader.__next__()
                # print(dssim_row)
                bg_diffs.append(int(row[3]) - int(dssim_row[3]))

                if int(row[6]) == 100000:
                    new_lev = 26
                else:
                    new_lev = int(row[6])

                if int(dssim_row[6]) == 100000 or int(dssim_row[6]) == -1:
                    old_lev = 26
                else:
                    old_lev = int(dssim_row[6])

                zfp_diffs.append(int((new_lev - old_lev)/2))

                v.append(row[0])
                time.append(row[2])

                ideal_bg = ast.literal_eval(row[9])
                ideal_zfp = ast.literal_eval(row[10])


    newdf = pd.DataFrame()
    newdf["bg_diff"] = bg_diffs
    newdf["zfp_diff"] = zfp_diffs
    newdf["var"] = v
    newdf["time"] = time

    # ax = plt.figure(figsize=(16, 10)).add_subplot(111)
    # plt.tick_params(
    #     axis='x',  # changes apply to the x-axis
    #     which='both',  # both major and minor ticks are affected
    #     bottom=False)


    #newdf.hist(["bg_diff", "zfp_diff"], title=f"Increase in compression level using all metrics", ylabel="Count",
    #           xlabel="Difference", ax=ax, kind='bar', legend=False)

    newdf.hist(["bg_diff", "zfp_diff"], bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
    plt.show()


    print(bg_diffs)
    print(zfp_diffs)