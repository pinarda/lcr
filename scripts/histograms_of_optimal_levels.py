# requires: daily_optimal_slices.csv and monthly_optimal_slices.csv.

import csv
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
mpl.use( 'tkagg' )

def hist_plotter(v, freq, csvfile, tags=True):
    df = pd.read_csv(csvfile)
    vdf = df[df["variable"]==v]
    #vdf=df

    a=vdf["best_alg"].to_list()
    l=vdf["best_level"].to_list()
    r=vdf["best_ratio"].to_list()
    rf = [float(x) for x in r]
    r = ["%.2f" % i for i in rf]

    new_list={}
    for i in range(0, len(a)):
        if f"{a[i]} {l[i]}" in new_list.keys():
            new_list[f"{a[i]} {l[i]}"] += 1
        else:
            new_list[f"{a[i]} {l[i]}"] = 0


    new_list = dict(sorted(new_list.items()))

    ratio_list = {}
    for i in range(0, len(a)):
        if f"{a[i]} {l[i]}" in ratio_list.keys():
            continue
        else:
            ratio_list[f"{a[i]} {l[i]}"] = r[i]

    ratio_list = dict(sorted(ratio_list.items()))

    level_list = {}
    for i in range(0, len(a)):
        if f"{a[i]} {l[i]}" in level_list.keys():
            continue
        else:
            level_list[f"{a[i]} {l[i]}"] = l[i]

    level_list = dict(sorted(level_list.items()))

    alg_list = set(a)
    algs = []
    cur_alg = None
    index=-1
    empty_indices = []
    for element in sorted(new_list.keys())[::-1]:
        index += 1
        for alg in alg_list:
            if alg in element:
                if cur_alg:
                    if alg != cur_alg:
                        empty_indices.append(index)
                        cur_alg = alg
                else:
                    cur_alg = alg
                algs.append(alg)

    newdf = pd.DataFrame()
    newdf["compression"] = sorted(new_list.keys())[::-1]
    newdf["algs"] = algs
    newdf["counts"] = sorted(new_list.values())[::-1]
    newdf.sort_values(by=["algs", "counts"], ascending=False, inplace=True)
    newdf = newdf.reset_index(drop=True)
    for index in empty_indices:
        newdf.loc[newdf.shape[0]] = ["", "", 0]
        # Move target row to first element of list.
        idx = list(newdf.index[0:index]) + [len(newdf)-1] + list(newdf.index[index:len(newdf)-1])
        newdf = newdf.reindex(idx)



    ax = plt.figure(figsize=(16, 10)).add_subplot(111)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False)
    newdf.plot("algs", ["counts"], title=f"Best Compression Histogram for {v} ({freq})", ylabel="Count",
               xlabel="Compression Settings", ax=ax, kind='bar', legend=False)

    rects = ax.patches

    # Make some labels
    labels = []
    for i in newdf["compression"]:
        if i:
            labels.append(f"Lev: {level_list[i]} Ratio: {ratio_list[i]}")
        else:
            labels.append("")

    for rect, label in zip(rects, labels):
        height = rect.get_height() + 0.5
        ax.text(
            rect.get_x() + rect.get_width() / 2, height, label, ha="center", va="bottom"
        )

    #ax.legend(loc='upper right', ncol=1)
    plt.savefig(f"../slice_hists/{v}{freq}")
    #plt.show()









if __name__ == "__main__":
     for v in ["FLNS", "FLNT", "FSNS", "FSNT", "LHFLX",
               "PRECC", "PRECL", "PS", "QFLX", "SHFLX", "TMQ", "TS"]:
         hist_plotter(v, "monthly", "../data/monthly_optimal_slices.csv")
    # for v in ["FLUT", "LHFLX", "PRECT", "TAUX", "TS", "Z500"]:
    #     hist_plotter(v, "daily", "../data/daily_optimal_slices.csv")
    # for v in ["FLUT", "LHFLX", "PRECT", "TAUX", "TS", "Z500"]:
    #  hist_plotter(v, "daily", "../data/daily_optimal_slices.csv")