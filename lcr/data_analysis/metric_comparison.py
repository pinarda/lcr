import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import ast
import numpy as np
mpl.use( 'tkagg' )

if __name__ == "__main__":
    csvfilename = "../../data/daily_zfp_bg_sz_comp_slices.csv"
    # csvfilename = "../data/daily_zfp_bg_sz_comp_slices_alternate.csv"
    # dssim_csvfilename = "../data/daily_zfp_bg_sz_comp_slices_dssim.csv"
    with open(csvfilename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            reader.__next__()
            bg_diffs = []
            zfp_diffs = []
            sz_diffs = []
            v = []
            time = []
            toughest_metric = []
            easiest_metric = []
            toughest_metric_zfp = []
            easiest_metric_zfp = []
            toughest_metric_sz = []
            easiest_metric_sz = []
            i=0
            for row in reader:
                v.append(row[0])
                time.append(row[2])

                ideal_zfp = ast.literal_eval(row[13])
                if int(ideal_zfp[0]) == 100000 or int(ideal_zfp[0]) == -1:
                    old_lev = 26
                else:
                    old_lev = int(ideal_zfp[0])
                if int(row[6]) == 100000:
                    new_lev = 26
                else:
                    new_lev = int(row[6])
                zfp_diffs.append(int((new_lev - old_lev)/2))

                ideal_bg = ast.literal_eval(row[12])
                bg_diffs.append(max(ideal_bg) - ideal_bg[0])

                ideal_sz = ast.literal_eval(row[14])
                if int(ideal_sz[0]) == 100000 or int(ideal_sz[0]) == -1:
                    old_lev = 0
                elif float(ideal_sz[0]) == 0.1:
                    old_lev = 1
                elif float(ideal_sz[0]) == 0.05:
                    old_lev = 2
                elif float(ideal_sz[0]) == 0.01:
                    old_lev = 3
                elif float(ideal_sz[0]) == 0.005:
                    old_lev = 4
                elif float(ideal_sz[0]) == 0.001:
                    old_lev = 5
                elif float(ideal_sz[0]) == 0.0005:
                    old_lev = 6
                elif float(ideal_sz[0]) == 0.0001:
                    old_lev = 7
                elif float(ideal_sz[0]) == 5e-05:
                    old_lev = 8
                elif float(ideal_sz[0]) == 1e-05:
                    old_lev = 9
                elif float(ideal_sz[0]) == 5e-06:
                    old_lev = 10
                elif float(ideal_sz[0]) == 1e-06:
                    old_lev = 11

                if float(row[9]) == 100000:
                    new_lev = 0
                elif float(row[9]) == 0.1:
                    new_lev = 1
                elif float(row[9]) == 0.05:
                    new_lev = 2
                elif float(row[9]) == 0.01:
                    new_lev = 3
                elif float(row[9]) == 0.005:
                    new_lev = 4
                elif float(row[9]) == 0.001:
                    new_lev = 5
                elif float(row[9]) == 0.0005:
                    new_lev = 6
                elif float(row[9]) == 0.0001:
                    new_lev = 7
                elif float(row[9]) == 5e-05:
                    new_lev = 8
                elif float(row[9]) == 1e-05:
                    new_lev = 9
                elif float(row[9]) == 5e-06:
                    new_lev = 10
                elif float(row[9]) == 1e-06:
                    new_lev = 11
                sz_diffs.append(int((new_lev - old_lev)))

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

                min_ideal_bg = min(ideal_bg)
                mins = [i for i, j in enumerate(ideal_bg) if j == min_ideal_bg]

                easiest = []
                if 0 in mins:
                    easiest.append("DSSIM")
                if 1 in mins:
                    easiest.append("KS P-value")
                if 2 in mins:
                    easiest.append("Spatial Relative Error")
                if 3 in mins:
                    easiest.append("Max Relative Error")
                if 4 in mins:
                    easiest.append("Pearson Correlation Coefficient")
                easiest_metric.append(easiest)

                ideal_zfp = ast.literal_eval(row[13])
                max_ideal_zfp = max(ideal_zfp)
                maxes_zfp = [i for i, j in enumerate(ideal_zfp) if j == max_ideal_zfp]

                toughest_zfp = []
                if 0 in maxes_zfp:
                    toughest_zfp.append("DSSIM")
                if 1 in maxes_zfp:
                    toughest_zfp.append("KS P-value")
                if 2 in maxes_zfp:
                    toughest_zfp.append("Spatial Relative Error")
                if 3 in maxes_zfp:
                    toughest_zfp.append("Max Relative Error")
                if 4 in maxes_zfp:
                    toughest.append("Pearson Correlation Coefficient")
                toughest_metric_zfp.append(toughest_zfp)

                min_ideal_zfp = min(ideal_zfp)
                mins_zfp = [i for i, j in enumerate(ideal_zfp) if j == min_ideal_zfp]

                easiest_zfp = []
                if 0 in mins_zfp:
                    easiest_zfp.append("DSSIM")
                if 1 in mins_zfp:
                    easiest_zfp.append("KS P-value")
                if 2 in mins_zfp:
                    easiest_zfp.append("Spatial Relative Error")
                if 3 in mins_zfp:
                    easiest_zfp.append("Max Relative Error")
                if 4 in mins_zfp:
                    easiest_zfp.append("Pearson Correlation Coefficient")
                easiest_metric_zfp.append(easiest_zfp)

                ideal_sz = ast.literal_eval(row[14])
                max_ideal_sz = max(ideal_sz)
                maxes_sz = [i for i, j in enumerate(ideal_sz) if j == max_ideal_sz]

                toughest_sz = []
                if 0 in maxes_sz:
                    toughest_sz.append("DSSIM")
                if 1 in maxes_sz:
                    toughest_sz.append("KS P-value")
                if 2 in maxes_sz:
                    toughest_sz.append("Spatial Relative Error")
                if 3 in maxes_sz:
                    toughest_sz.append("Max Relative Error")
                if 4 in maxes_sz:
                    toughest.append("Pearson Correlation Coefficient")
                toughest_metric_sz.append(toughest_sz)

                min_ideal_sz = min(ideal_sz)
                mins_sz = [i for i, j in enumerate(ideal_sz) if j == min_ideal_sz]

                easiest_sz = []
                if 0 in mins_sz:
                    easiest_sz.append("DSSIM")
                if 1 in mins_sz:
                    easiest_sz.append("KS P-value")
                if 2 in mins_sz:
                    easiest_sz.append("Spatial Relative Error")
                if 3 in mins_sz:
                    easiest_zfp.append("Max Relative Error")
                if 4 in mins_sz:
                    easiest_sz.append("Pearson Correlation Coefficient")
                easiest_metric_sz.append(easiest_sz)

                i=i+1




    newdf = pd.DataFrame()
    newdf["bg_diff"] = bg_diffs
    newdf["zfp_diff"] = zfp_diffs
    newdf["sz_diff"] = sz_diffs
    newdf["vars"] = v
    newdf["time"] = time
    newdf["toughest"] = toughest_metric
    newdf["easiest"] = easiest_metric
    newdf["toughest_zfp"] = toughest_metric_zfp
    newdf["easiest_zfp"] = easiest_metric_zfp
    newdf["toughest_sz"] = toughest_metric_sz
    newdf["easiest_sz"] = easiest_metric_sz

    # ax = plt.figure(figsize=(16, 10)).add_subplot(111)
    # plt.tick_params(
    #     axis='x',  # changes apply to the x-axis
    #     which='both',  # both major and minor ticks are affected
    #     bottom=False)


    #newdf.hist(["bg_diff", "zfp_diff"], title=f"Increase in compression level using all metrics", ylabel="Count",
    #           xlabel="Difference", ax=ax, kind='bar', legend=False)

    # newdf.hist(["bg_diff", "zfp_diff"], bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
    # plt.show()

    np.save("../../data/toughest_bg.npy", newdf["toughest"])
    np.save("../../data/toughest_zfp.npy", newdf["toughest_zfp"])
    np.save("../../data/toughest_sz.npy", newdf["toughest_sz"])

    large_diff_df = newdf[newdf["bg_diff"]>=2]

    (large_diff_df.vars.value_counts() / len(large_diff_df)).sort_index().plot.bar(title=f"Fraction of variable showing increase of at least 2 (bg) levels")
    plt.ylabel("Fraction")
    plt.ylim(top=0.5)

    plt.xticks(rotation=90)
    plt.xlabel("Daily Variable")
    plt.show()

    large_diff_df_zfp = newdf[newdf["zfp_diff"]>=2]

    (large_diff_df_zfp.vars.value_counts() / len(large_diff_df_zfp)).sort_index().plot.bar(title=f"Fraction of variable showing increase of at least 2 (zfp) levels")
    plt.ylabel("Fraction")
    plt.ylim(top=0.5)
    plt.xticks(rotation=90)
    plt.xlabel("Daily Variable")
    plt.show()

    large_diff_df_sz = newdf[newdf["sz_diff"] >= 2]

    (large_diff_df_sz.vars.value_counts() / len(large_diff_df_sz)).sort_index().plot.bar(
        title=f"Fraction of variable showing increase of at least 2 (sz) levels")
    plt.ylabel("Fraction")
    plt.ylim(top=0.5)
    plt.xticks(rotation=90)
    plt.xlabel("Daily Variable")
    plt.show()

    (newdf.bg_diff.value_counts()/len(newdf)).sort_index().plot.bar(title=f"Increase in bg compression level using all metrics")
    plt.ylabel("Fraction")
    plt.xticks(rotation=90)
    plt.xlabel("Level Increase")
    plt.ylim(top=1)
    plt.show()

    (newdf.zfp_diff.value_counts()/len(newdf)).sort_index().plot.bar(title=f"Increase in zfp compression level using all metrics")
    plt.ylabel("Fraction")
    plt.xticks(rotation=90)
    plt.xlabel("Level Increase")
    plt.ylim(top=1)
    plt.show()

    (newdf.sz_diff.value_counts()/len(newdf)).sort_index().plot.bar(title=f"Increase in sz compression level using all metrics")
    plt.ylabel("Fraction")
    plt.xticks(rotation=90)
    plt.xlabel("Level Increase")
    plt.ylim(top=1)
    plt.show()

    anothernewdf = pd.DataFrame()
    toughest_list = []
    for i in range(0, len(newdf)):
        for j in range(0, len(newdf.toughest[i])):
            toughest_list.append(newdf.toughest[i][j])
    anothernewdf["toughest"] = toughest_list
    (anothernewdf.toughest.value_counts()/len(newdf)).plot.bar(title=f"Metric(s) requiring most conservative threshold (bg)")
    plt.ylabel("Fraction")
    plt.ylim(top=1)
    plt.xlabel("Metric")
    plt.xticks(rotation=0)
    plt.show()

    anothernewdf = pd.DataFrame()
    toughest_list = []
    for i in range(0, len(newdf)):
        for j in range(0, len(newdf.toughest_zfp[i])):
            toughest_list.append(newdf.toughest_zfp[i][j])
    anothernewdf["toughest"] = toughest_list
    (anothernewdf.toughest.value_counts() / len(newdf)).plot.bar(
        title=f"Metric(s) requiring most conservative threshold (zfp)")
    plt.ylabel("Fraction")
    plt.ylim(top=1)
    plt.xlabel("Metric")
    plt.xticks(rotation=0)
    plt.show()

    anothernewdf = pd.DataFrame()
    toughest_list = []
    for i in range(0, len(newdf)):
        for j in range(0, len(newdf.toughest_sz[i])):
            toughest_list.append(newdf.toughest_sz[i][j])
    anothernewdf["toughest"] = toughest_list
    (anothernewdf.toughest.value_counts() / len(newdf)).plot.bar(
        title=f"Metric(s) requiring most conservative threshold (sz)")
    plt.ylabel("Fraction")
    plt.ylim(top=1)
    plt.xlabel("Metric")
    plt.xticks(rotation=0)
    plt.show()

    mynewdf = pd.DataFrame()
    easiest_list = []
    for i in range(0, len(newdf)):
        for j in range(0, len(newdf.easiest[i])):
            easiest_list.append(newdf.easiest[i][j])
    mynewdf["easiest"] = easiest_list
    (mynewdf.easiest.value_counts()/len(newdf)).plot.bar(title=f"Metric(s) requiring least conservative threshold (bg)")
    plt.ylabel("Fraction")
    plt.ylim(top=1)
    plt.xticks(rotation=0)
    plt.xlabel("Metric")
    plt.show()

    mynewdf = pd.DataFrame()
    easiest_list = []
    for i in range(0, len(newdf)):
        for j in range(0, len(newdf.easiest_zfp[i])):
            easiest_list.append(newdf.easiest_zfp[i][j])
    mynewdf["easiest"] = easiest_list
    (mynewdf.easiest.value_counts() / len(newdf)).plot.bar(title=f"Metric(s) requiring least conservative threshold (zfp)")
    plt.ylabel("Fraction")
    plt.ylim(top=1)
    plt.xticks(rotation=0)
    plt.xlabel("Metric")
    plt.show()

    mynewdf = pd.DataFrame()
    easiest_list = []
    for i in range(0, len(newdf)):
        for j in range(0, len(newdf.easiest_sz[i])):
            easiest_list.append(newdf.easiest_sz[i][j])
    mynewdf["easiest"] = easiest_list
    (mynewdf.easiest.value_counts() / len(newdf)).plot.bar(title=f"Metric(s) requiring least conservative threshold (sz)")
    plt.ylabel("Fraction")
    plt.ylim(top=1)
    plt.xticks(rotation=0)
    plt.xlabel("Metric")
    plt.show()


    print(bg_diffs)
    print(zfp_diffs)
    print(sz_diffs)