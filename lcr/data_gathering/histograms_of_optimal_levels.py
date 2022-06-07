"""
requires: daily_optimal_slices.csv and monthly_optimal_slices.csv. (see compare_algorithms.csv)
used for: daily/monthly_labels.csv (for create_dataframe.csv)
"""

import sys
from pathlib import Path
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import lcr_global_vars
from os.path import exists
mpl.use( 'tkagg' )

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

mpl.rc('font', **font)

def omit_by(dct, predicate=lambda x: x==0):
    return {k: v for k, v in dct.items() if not predicate(v)}



def hist_plotter(v, freq, csvfile, tags=True):
    df = pd.read_csv(csvfile)
    vdf = df[df["variable"]==v]
    #vdf=df

    adf=df["best_alg"].to_list()
    ldf=df["best_level"].to_list()
    a=vdf["best_alg"].to_list()
    l=vdf["best_level"].to_list()
    r=vdf["best_ratio"].to_list()
    rf = [float(x) for x in r]
    r = ["%.2f" % i for i in rf]

    # yikes
    new_list={}
    for i in range(0, len(adf)):
        new_list[f"{adf[i]} {ldf[i]}"] = 0

    for i in range(0, len(a)):
        if f"{a[i]} {l[i]}" in new_list.keys():
            new_list[f"{a[i]} {l[i]}"] += 1
        else:
            new_list[f"{a[i]} {l[i]}"] = 0


    new_list = dict(sorted(new_list.items()))

    ratio_list = {}
    n={}
    rfloat = [float(x) for x in r]
    # yikes
    for i in range(0, len(adf)):
        ratio_list[f"{adf[i]} {ldf[i]}"] = 0
        n[f"{adf[i]} {ldf[i]}"] = 0

    for i in range(0, len(a)):
        ratio_list[f"{a[i]} {l[i]}"] = ratio_list[f"{a[i]} {l[i]}"] + float(r[i])
        n[f"{a[i]} {l[i]}"] = n[f"{a[i]} {l[i]}"]+1

    for key in ratio_list.keys():
        if n[key] == 0:
            ratio_list[key] = str(1)
        else:
            ratio_list[key] = str(ratio_list[key]/n[key])

    ratio_list = dict(sorted(ratio_list.items()))

    level_list = {}
    # yikes
    for i in range(0, len(adf)):
        level_list[f"{adf[i]} {ldf[i]}"] = ldf[i]
    for i in range(0, len(a)):
        level_list[f"{a[i]} {l[i]}"] = l[i]

    level_list = dict(sorted(level_list.items()))

    alg_list = set(adf)
    algs = []
    cur_alg = None
    index=-1
    empty_count = 0
    empty_indices = []
    for element in sorted(new_list.keys())[::-1]:
        index += 1
        for alg in alg_list:
            if alg in element:
                if cur_alg:
                    if alg != cur_alg:
                        empty_indices.append(index + empty_count)
                        empty_count += 1
                        cur_alg = alg
                else:
                    cur_alg = alg
                algs.append(alg)

    newdf = pd.DataFrame()
    newdf["compression"] = sorted(new_list.keys())[::-1]
    newdf["algs"] = algs

    possible_algs = ["zfp", "sz", "bg", "z_hdf5"]
    colors = []

    actual_t = omit_by(new_list)
    my_algs = []
    for actual_alg in actual_t:
        s = list(actual_alg == newdf["compression"])
        my_algs.append(newdf["algs"][[i for i in range(len(s)) if s[i] is True][0]])

    for alg in algs:
        if alg == possible_algs[0]:
            colors.append("blue")
        elif alg == possible_algs[1]:
            colors.append("green")
        elif alg == possible_algs[2]:
            colors.append("orange")
        elif alg == possible_algs[3]:
            colors.append("cyan")
        else:
            colors.append("red")

    newdf["color"] = colors

    newdf["counts"] = list(new_list.values())[::-1]
    #newdf.sort_values(by=["algs", "counts"], ascending=False, inplace=True)
    newdf = newdf.reset_index(drop=True)
    n=0
    for index in empty_indices:
        newdf.loc[newdf.shape[0]] = [" " * n, "", "blue", 0]
        n=n+1
        # Move target row to first element of list.
        idx = list(newdf.index[0:index]) + [len(newdf)-1] + list(newdf.index[index:len(newdf)-1])
        newdf = newdf.reindex(idx)

    #moves lossless to end
    copydf = newdf
    copydf.compression[copydf.compression == "zfp 100000"] = 'Lossless'
    posindex = copydf.index[copydf.compression == "Lossless"].to_list()
    negindex = copydf.index[copydf.compression != "Lossless"].to_list()
    copydf = copydf.reindex(posindex + negindex)


    ax = plt.figure(figsize=(16, 10)).add_subplot(111)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False)

#    newdf.plot("algs", ["counts"], title=f"Best Compression Histogram for {v} ({freq})", ylabel="Count",
#              xlabel="Compression Settings", ax=ax, kind='bar', legend=False, color=colors)
    plt.bar(x=list(copydf["compression"])[::-1], height=list(copydf["counts"])[::-1], color = list(copydf["color"])[::-1])
    plt.title(f"Best Compression Histogram for {v} ({freq})")
    plt.xlabel("Compression Settings")
    plt.xticks(plt.xticks()[0], labels=newdf["algs"][::-1])
    plt.ylabel("Count")
    plt.ylim(top=2000)

    rects = ax.patches

    # Make some labels
    labels = []
    for i in copydf["compression"][::-1]:
        if i == "Lossless" and int(copydf.loc[copydf['compression'] == "Lossless"].counts) != 0:
            labels.append(f"Lossless")
        elif i in list(actual_t.keys()):
            labels.append(f"Lev: {level_list[i]} Ratio: {float(ratio_list[i]):.1f}")
        #elif i:
        #    labels.append(f"Lev: {level_list[i]} Ratio: N/A")
        else:
            labels.append("")

    for rect, label in zip(rects, labels):
        height = rect.get_height() + 0.5
        ax.text(
            rect.get_x() + rect.get_width() / 2, height, label, ha="center", va="bottom", fontsize=8
        )

    hatches = ''
    for color in list(copydf["color"][::-1]):
        if color == "blue":
            hatches = hatches + "x"
        elif color == "red":
            hatches = hatches + "+"
        elif color == "orange":
            hatches = hatches + "/"
        elif color == "cyan":
            hatches = hatches + "|"
        else:
            hatches = hatches + "O"

    for bar, hatch in zip(rects, hatches):
        bar.set_hatch(hatch)

    #ax.legend(loc='upper right', ncol=1)
    plt.savefig(f"../slice_hists/{v}{freq}.png")
    #plt.show()

def save_labels(v, freq, csvfile, tags=True):
    df = pd.read_csv(csvfile)
    vdf = df[df["variable"]==v]
    #vdf=df

    adf=df["best_alg"].to_list()
    ldf=df["best_level"].to_list()
    a=vdf["best_alg"].to_list()
    l=vdf["best_level"].to_list()
    r=vdf["best_ratio"].to_list()
    v=vdf["variable"].to_list()
    rf = [float(x) for x in r]
    r = ["%.2f" % i for i in rf]

    # yikes
    new_list={}
    for i in range(0, len(adf)):
        new_list[f"{adf[i]} {ldf[i]}"] = 0

    for i in range(0, len(a)):
        if f"{a[i]} {l[i]}" in new_list.keys():
            new_list[f"{a[i]} {l[i]}"] += 1
        else:
            new_list[f"{a[i]} {l[i]}"] = 0


    new_list = dict(sorted(new_list.items()))

    ratio_list = {}
    # yikes
    for i in range(0, len(adf)):
        ratio_list[f"{adf[i]} {ldf[i]}"] = 0

    for i in range(0, len(a)):
        ratio_list[f"{a[i]} {l[i]}"] = r[i]

    ratio_list = dict(sorted(ratio_list.items()))

    level_list = {}
    # yikes
    for i in range(0, len(adf)):
        level_list[f"{adf[i]} {ldf[i]}"] = ldf[i]
    for i in range(0, len(a)):
        level_list[f"{a[i]} {l[i]}"] = l[i]

    level_list = dict(sorted(level_list.items()))

    alg_list = set(adf)
    algs = []
    cur_alg = None
    index=-1
    empty_count = 0
    empty_indices = []
    for element in sorted(new_list.keys())[::-1]:
        index += 1
        for alg in alg_list:
            if alg in element:
                if cur_alg:
                    if alg != cur_alg:
                        empty_indices.append(index + empty_count)
                        empty_count += 1
                        cur_alg = alg
                else:
                    cur_alg = alg
                algs.append(alg)

    newdf = pd.DataFrame()
    newdf["levels"] = l
    newdf["algs"] = a
    newdf["ratios"] = r
    newdf["variable"] = v
    # At the moment it is assumed that the time slices are in order here.
    newdf["times"] = list(range(0,2190))

    fileloc = f"../data/{freq}_labels.csv"
    file_exists = exists(fileloc)
    if file_exists:
        newdf.to_csv(fileloc, mode="a", header=False, index=False)
    else:
        newdf.to_csv(fileloc, mode="a", header=True, index=False)


if __name__ == "__main__":
    # create histograms
    # for v in ["FLNS", "FLNT", "FSNS", "FSNT", "LHFLX",
    #            "PRECC", "PRECL", "PS", "QFLX", "SHFLX", "TMQ", "TS"]:
    #      hist_plotter(v, "monthly", "../data/monthly_optimal_slices.csv")
    # for v in lcr_global_vars.daily_vars:
    #      hist_plotter(v, "daily", "../data/daily_optimal_slices.csv")

    # just save the dataframe as labels
    # for v in lcr_global_vars.monthly_vars:
    #     save_labels(v, "monthly", "../data/monthly_optimal_slices.csv")

    #USE THIS FOR MAKING DAILY/MONTHLY LABELS
    for v in lcr_global_vars.daily_vars:
       save_labels(v, "daily", "../data/daily_optimal_slices.csv")