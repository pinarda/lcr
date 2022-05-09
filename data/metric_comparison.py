import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import ast
mpl.use( 'tkagg' )

if __name__ == "__main__":
    csvfilename = "../data/daily_zfp_bg_sz_comp_slices.csv"
    # csvfilename = "../data/daily_zfp_bg_sz_comp_slices_alternate.csv"
    # dssim_csvfilename = "../data/daily_zfp_bg_sz_comp_slices_dssim.csv"
    with open(csvfilename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            reader.__next__()
            bg_diffs = []
            zfp_diffs = []
            v = []
            time = []
            toughest_metric = []
            i=0
            for row in reader:
                v.append(row[0])
                time.append(row[2])

                ideal_zfp = ast.literal_eval(row[10])
                if int(ideal_zfp[0]) == 100000 or int(ideal_zfp[0]) == -1:
                    old_lev = 26
                else:
                    old_lev = int(ideal_zfp[0])
                if int(row[6]) == 100000:
                    new_lev = 26
                else:
                    new_lev = int(row[6])
                zfp_diffs.append(int((new_lev - old_lev)/2))

                ideal_bg = ast.literal_eval(row[9])
                bg_diffs.append(max(ideal_bg) - ideal_bg[0])
                max_ideal_bg = max(ideal_bg)
                maxes = [i for i, j in enumerate(ideal_bg) if j == max_ideal_bg]

                toughest = []
                if 0 in maxes:
                    toughest.append("DSSIM")
                if 1 in maxes:
                    toughest.append("KS P-value")
                if 2 in maxes:
                    toughest.append("Spatial Relative Error")
                if 3 in maxes:
                    toughest.append("Max Relative Error")
                if 4 in maxes:
                    toughest.append("Pearson Correlation Coefficient")
                toughest_metric.append(toughest)

                ideal_zfp = ast.literal_eval(row[10])
                i=i+1




    newdf = pd.DataFrame()
    newdf["bg_diff"] = bg_diffs
    newdf["zfp_diff"] = zfp_diffs
    newdf["var"] = v
    newdf["time"] = time
    newdf["toughest"] = toughest_metric

    # ax = plt.figure(figsize=(16, 10)).add_subplot(111)
    # plt.tick_params(
    #     axis='x',  # changes apply to the x-axis
    #     which='both',  # both major and minor ticks are affected
    #     bottom=False)


    #newdf.hist(["bg_diff", "zfp_diff"], title=f"Increase in compression level using all metrics", ylabel="Count",
    #           xlabel="Difference", ax=ax, kind='bar', legend=False)

    # newdf.hist(["bg_diff", "zfp_diff"], bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
    # plt.show()

    newdf.bg_diff.value_counts().sort_index().plot.bar(title=f"Increase in bg compression level using all metrics")
    plt.ylim(top=25000)
    plt.show()

    newdf.zfp_diff.value_counts().sort_index().plot.bar(title=f"Increase in zfp compression level using all metrics")
    plt.ylim(top=25000)
    plt.show()

    anothernewdf = pd.DataFrame()
    toughest_list = []
    for i in range(0, len(newdf)):
        for j in range(0, len(newdf.toughest[i])):
            toughest_list.append(newdf.toughest[i][j])
    anothernewdf["toughest"] = toughest_list
    anothernewdf.toughest.value_counts().plot.bar(title=f"Metric(s) requiring most conservative threshold")
    plt.ylim(top=25000)
    plt.xticks(rotation=0)
    plt.show()


    print(bg_diffs)
    print(zfp_diffs)