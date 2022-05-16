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
            easiest_metric = []
            toughest_metric_zfp = []
            easiest_metric_zfp = []
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

                ideal_zfp = ast.literal_eval(row[10])
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

                i=i+1




    newdf = pd.DataFrame()
    newdf["bg_diff"] = bg_diffs
    newdf["zfp_diff"] = zfp_diffs
    newdf["vars"] = v
    newdf["time"] = time
    newdf["toughest"] = toughest_metric
    newdf["easiest"] = easiest_metric
    newdf["toughest_zfp"] = toughest_metric_zfp
    newdf["easiest_zfp"] = easiest_metric_zfp

    # ax = plt.figure(figsize=(16, 10)).add_subplot(111)
    # plt.tick_params(
    #     axis='x',  # changes apply to the x-axis
    #     which='both',  # both major and minor ticks are affected
    #     bottom=False)


    #newdf.hist(["bg_diff", "zfp_diff"], title=f"Increase in compression level using all metrics", ylabel="Count",
    #           xlabel="Difference", ax=ax, kind='bar', legend=False)

    # newdf.hist(["bg_diff", "zfp_diff"], bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
    # plt.show()


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

    anothernewdf = pd.DataFrame()
    toughest_list = []
    for i in range(0, len(newdf)):
        for j in range(0, len(newdf.toughest[i])):
            toughest_list.append(newdf.toughest[i][j])
    anothernewdf["toughest"] = toughest_list
    (anothernewdf.toughest.value_counts()/len(newdf)).plot.bar(title=f"Metric(s) requiring most conservative threshold")
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
    (mynewdf.easiest.value_counts()/len(newdf)).plot.bar(title=f"Metric(s) requiring least conservative threshold")
    plt.ylabel("Fraction")
    plt.ylim(top=1)
    plt.xticks(rotation=0)
    plt.xlabel("Metric")
    plt.show()


    print(bg_diffs)
    print(zfp_diffs)