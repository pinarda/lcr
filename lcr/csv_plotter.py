import matplotlib as mpl
import tkinter
import matplotlib.pyplot as plt
import pandas as pd
mpl.use( 'tkagg' )



def plot_optimal_level_comparison():
    data_dir = "../data"
    for freq in ["daily", "monthly"]:
        df = pd.read_csv(f"{data_dir}/{freq}_zfp_bg_sz_comparison.csv")
        ax = plt.figure(figsize=(10, 6)).add_subplot(111)
        df.plot("variable", ["zfp_ratio", "bg_ratio", "sz_ratio"], title=f"Lossless/Lossy Compression Ratios ({freq})", ylabel="Ratio", ax=ax, kind='bar', legend=False)

        bars = ax.patches
        hatches = ''.join(h * len(df) for h in 'x/O.')

        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 0.9), ncol=1)
        plt.show()
        # plt.savefig(f"../plots/{freq}_comp_ratios.png")

def plot_optimal_ratios():
    data_dir = "../data"
    for freq in ["daily", "monthly"]:
        df = pd.read_csv(f"{data_dir}/{freq}_optimal_over_var.csv")
        ax = plt.figure(figsize=(10, 6)).add_subplot(111)
        df.plot("variable", ["best_ratio"], title=f"Best Lossless/Lossy Compression Ratio ({freq})", ylabel="Ratio", ax=ax, kind='bar', legend=False, ylim=[0,2.2])

        rects = ax.patches

        # Make some labels
        labels = df["best_alg"]

        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2, height, label, ha="center", va="bottom"
            )

        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 0.9), ncol=1)
        plt.show()
        # plt.savefig(f"../plots/{freq}_comp_ratios.png")


if __name__ == "__main__":
    #plot_optimal_level_comparison()
    plot_optimal_ratios()

    #data_dir = "../manual_data"
    #for comp in ["zfp", "sz", "bg"]:
    #    df = pd.read_csv(f"{data_dir}/{comp}hist.csv")
    #    print(df)


