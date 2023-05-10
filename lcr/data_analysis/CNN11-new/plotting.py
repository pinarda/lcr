import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.use('TKAgg')

def generate_performance_plots(x, errors, dssims, preds, legend, j, plotdir, name, storageloc):
    # create a plot of the errors, errors1, and errors3 vs. time
    num_colors = len(dssims)
    i=0
    for error in errors:
        ibin = bin(num_colors)[2:].zfill(3)
        ibin = bin(i)[2:].zfill(3)
        #plt.semilogy(x, error, color=(int(ibin[0]), int(ibin[1]), int(ibin[2])))

        # error is a list of data that should be plotted in a single box plot (one for each compression level)
        # each error box plot should be drawn on the same plot
        # the x-axis should be the number of time steps used
        # the y-axis should be the prediction error boxplot for a single timestep (on a log scale)

        # first, reorder errors so that we have a list of lists, where each list is the errors for a single timestep
        # then, plot each list as a boxplot

        # make sure the y-axis is on a log scale

        p = plt.boxplot(np.log10(np.array(error)), positions=x, patch_artist=True)
        for box in p['boxes']:
            box.set_facecolor((int(ibin[0]), int(ibin[1]), int(ibin[2])))



        i += 1
    l = []
    for item in legend:
        l.append(item + " error")
    plt.xlabel("Time Slices Used")
    plt.ylabel("Prediction error for single timestep (log10 scale)")
    plt.title("DSSIM prediction error vs. Time Slices Used")

    #plt.legend(l, loc='upper right', bbox_to_anchor=(1, 0.5))
    ax = plt.gca()



    # leg = ax.get_legend()
    # i=0
    # for item in legend:
    #     ibin = bin(i)[2:].zfill(3)
    #     leg.legendHandles[i].set_color((int(ibin[0]), int(ibin[1]), int(ibin[2])))
    #     i += 1

    plt.savefig(f"{storageloc}error_all_vs_time_{name}.png")
    plt.clf()
    # create a plot of the average dssim vs. time, and overlay the average prediction

    i=0
    for dssim in dssims:
        ibin = bin(num_colors)[2:].zfill(3)
        ibin = bin(i)[2:].zfill(3)
        # the dssim will always be the same, the way it is set up now, so we don't need to make a boxplot
        plt.semilogy(x, dssim[0], color=(int(ibin[0]), int(ibin[1]), int(ibin[2])))
        i += 1
    i=0
    for pred in preds:
        ibin = bin(num_colors)[2:].zfill(3)
        ibin = bin(i)[2:].zfill(3)
        #plt.semilogy(x, pred, linestyle='dashed', color=(int(ibin[0]), int(ibin[1]), int(ibin[2])))
        p = plt.boxplot(np.array(pred), positions=x, patch_artist=True)
        for box in p['boxes']:
            box.set_facecolor((int(ibin[0]), int(ibin[1]), int(ibin[2])))
        i += 1
    l = []
    for item in legend:
        l.append(item + " DSSIM")
    for item in legend:
        l.append(item + " Prediction")
   # plt.legend(l, loc='upper right', bbox_to_anchor=(1, 0.5))
    plt.xlabel("Time Slices Used")
    plt.ylabel("Average DSSIM/Prediction")
    plt.title("Average DSSIM/Prediction vs. Time Slices Used")
    plt.savefig(f"{storageloc}dssim_all_vs_time_{name}.png")
    plt.clf()

    # # crreate a plot of the difference between the average dssim and the average prediction vs. time
    # i=0
    # for dssim, pred in zip(dssims, preds):
    #     ibin = bin(num_colors)[2:].zfill(3)
    #     ibin = bin(i)[2:].zfill(3)
    #     y = np.array(dssim) - np.array(pred)
    #     #plt.semilogy(x, y, color=(int(ibin[0]), int(ibin[1]), int(ibin[2])))
    #     # log the y values as long as they are positive and not nan
    #     for j in range(len(y)):
    #         if np.all(y[j] > 0) and not np.any(np.isnan(y[j])):
    #             y[j] = np.log10(y[j])
    #         else:
    #             y[j] = 0
    #     p = plt.semilogy(test_slices, np.array(y).squeeze().mean(), color=(int(ibin[0]), int(ibin[1]), int(ibin[2])))
    #     # for box in p['boxes']:
    #     #     box.set_facecolor((int(ibin[0]), int(ibin[1]), int(ibin[2])))
    #     i += 1
    # # l = []
    # # lines = []
    # # i=0
    # # for item in legend:
    # #     ibin = bin(i)[2:].zfill(3)
    # #     l.append(item + " tolerance")
    # #     lines.append(Line2D([0], [0], color=(int(ibin[0]), int(ibin[1]), int(ibin[2])), lw=4))
    # #     i=i+1
    # # plt.legend(lines, l, loc='upper right', bbox_to_anchor=(1, 0.5))
    # plt.xlabel("Time Slices Used")
    # plt.ylabel("Average DSSIM - Average Prediction")
    # plt.title("Average DSSIM - Average Prediction vs. Time Slices Used")
    # plt.savefig(f"{storageloc}dssim-pred_all_vs_time_{name}.png")
    # plt.clf()