import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import argparse
import json
import os
import pandas as pd
import math
import matplotlib as mpl
import matplotlib.patches as mpatches

def return_first_existing(paths):
    """
    Try each path in *paths* (a list/tuple of strings) and return
    np.load(path, **np_load_kwargs) for the first one that exists.
    Raises FileNotFoundError if none exist.
    """
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("None of the candidate files exist:\n  " +
                            "\n  ".join(paths))
def load_first_existing(paths, **np_load_kwargs):
    """
    Try each path in *paths* (a list/tuple of strings) and return
    np.load(path, **np_load_kwargs) for the first one that exists.
    Raises FileNotFoundError if none exist.
    """
    for p in paths:
        if os.path.exists(p):
            return np.load(p, **np_load_kwargs)
    raise FileNotFoundError("None of the candidate files exist:\n  " +
                            "\n  ".join(paths))

def process_config(config_file):
    """Process a single configuration file and calculate F1 scores."""
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Extract parameters
    storageloc = config.get('StorageLoc')           # "./data/"
    times = config.get('Times')                     # [60]
    time = times[0]
    j = 0
    jobid = 0
    var_list = config.get('VarList')                # [["TS"], ["PRECT"]]
    rf_feature_list = config.get('RFFeatureList')   # Feature names for RF

    # Use var_list[0] for file names directly
    variable_file = var_list[0]
    # Use the actual string for display purposes
    variable_display = var_list[0][0] if isinstance(var_list[0], list) and var_list[0] else "Unknown"

    # Load data for CNN
    # test_labels_np_cnn = np.load(f"{storageloc}/test_labels_{j}{time}cnn{jobid}_{variable_file}.npy")
    # test_predictions_cnn = np.load(f"{storageloc}/test_predictions_{j}{time}cnn{jobid}_{variable_file}.npy")

    # Load data for RF
    # test_labels_np_rf = np.load(f"{storageloc}/test_labels_{j}{time}rf{jobid}_{variable_file}.npy")
    test_labels_np_rf = load_first_existing(
        [
            f"{storageloc}/test_labels_02000rf0_{variable_file}.npy",
            f"{storageloc}/test_labels_12000rf0_{variable_file}.npy",
        ],
        allow_pickle=True
    )
    # test_predictions_rf = np.load(f"{storageloc}/test_predictions_rf_{j}{time}rf{jobid}_{variable_file[0]}.npy")
    test_predictions_rf = load_first_existing(
        [
            f"{storageloc}/test_predictions_rf_42000rf0_{variable_file}.npy",
            f"{storageloc}/test_predictions_rf_32000rf0_{variable_file}.npy",
            f"{storageloc}/test_predictions_rf_22000rf0_{variable_file}.npy",
            f"{storageloc}/test_predictions_rf_12000rf0_{variable_file}.npy",
            f"{storageloc}/test_predictions_rf_02000rf0_{variable_file}.npy",
        ],
        allow_pickle=True
    )


    # Compute F1 scores
    # f1_weighted_cnn = f1_score(test_labels_np_cnn, test_predictions_cnn, average='weighted')
    # f1_macro_cnn = f1_score(test_labels_np_cnn, test_predictions_cnn, average='macro')
    f1_weighted_rf = f1_score(test_labels_np_rf, test_predictions_rf, average='weighted')
    f1_macro_rf = f1_score(test_labels_np_rf, test_predictions_rf, average='macro')



    # Load feature importances for RF
    # feature_importance_file = f"{storageloc}/feature_importances_rf_02000rf0_{variable_file[0]}.npy"
    feature_importance_file = return_first_existing(
        [
            f"{storageloc}/feature_importances_rf_42000rf0_{variable_file}.npy",
            f"{storageloc}/feature_importances_rf_32000rf0_{variable_file}.npy",
            f"{storageloc}/feature_importances_rf_22000rf0_{variable_file}.npy",
            f"{storageloc}/feature_importances_rf_12000rf0_{variable_file}.npy",
            f"{storageloc}/feature_importances_rf_02000rf0_{variable_file}.npy",
        ]
    )
    print(f"feature_importance_file: {feature_importance_file}")
    if os.path.exists(feature_importance_file):
        feature_importances = np.load(feature_importance_file)
        print("loading importances...")
    else:
        feature_importances = None


    compression_labels = {0: "zfp_p_16", 1: "zfp_p_22"}
    table_data = []

    for label, prediction in zip(test_labels_np_rf, test_predictions_rf):
        label_name = compression_labels.get(label, "Unknown")
        prediction_name = compression_labels.get(prediction, "Unknown")
        table_data.append((variable_file[0], label_name, f"{label_name}, {prediction_name}"))


    # Convert to DataFrame
    df = pd.DataFrame(table_data, columns=["Variable", "Compression Label", "Labels and Predictions"])
    # dupes = df[df.duplicated(subset=["Variable", "Compression Label"], keep=False)]
    # print(dupes.sort_values(["Variable", "Compression Label"]).head())
    df_unique = df.drop_duplicates(subset=["Variable", "Compression Label"], keep="first")
    df_pivot = df_unique.pivot(index="Variable", columns="Compression Label", values="Labels and Predictions").fillna("")

    # Print the pivot table
    print("Pivot Table:")
    print(df_pivot)

    # Format the table as LaTeX
    latex_table = df_pivot.to_latex(index=True, caption="Labels and Predictions for Compression Methods")

    # Save the LaTeX table to a file
    latex_file_path = f"{storageloc}/table_output.tex"
    with open(latex_file_path, "w") as f:
        f.write(latex_table)

    print(f"LaTeX table saved to {latex_file_path}")

    return variable_display, f1_weighted_rf, f1_macro_rf, feature_importances, rf_feature_list


def plot_f1_scores(var_list, rf_scores, metric_name, filename):
    """Create a bar plot for F1 scores."""
    x = np.arange(len(var_list))  # X-axis positions
    bar_width = 0.35  # Width of each bar

    fig, ax = plt.subplots(figsize=(10, 6))

    # CNN F1 scores
    # ax.bar(x - bar_width / 2, rf_scores, bar_width, label="CNN F1 Score", color="blue", edgecolor="black")

    # RF F1 scores
    ax.bar(x + bar_width / 2, rf_scores, bar_width, label="RF F1 Score", color="red", edgecolor="black")

    # Adding labels and legend
    ax.set_xlabel("Variables")
    ax.set_ylabel(f"{metric_name} F1 Score")
    ax.set_title(f"{metric_name} F1 Score Comparison for RF Models")
    ax.set_xticks(x)
    ax.set_xticklabels(var_list, rotation=45, ha='right')  # Rotate for better readability
    ax.legend()

    # Display and save the plot
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"{metric_name} F1 score plot saved to {filename}")


def plot_feature_importances(features, all_importances, variable_names, filename):
    """
    Create a grouped bar plot for feature importances across all variables.

    Parameters:
        features (list): List of feature names.
        all_importances (list of lists): 2D list where each inner list contains feature importances for a variable.
        variable_names (list): List of variable names corresponding to all_importances.
        filename (str): Path to save the plot.
    """
    if not all_importances or not features:
        print("No data for feature importance plot.")
        return

    #print features
    print(features)
    print(variable_names)
    print(all_importances)


    # Number of features and variables
    n_features = len(features)
    n_variables = len(all_importances)

    # X-axis positions
    x = np.arange(n_features)  # Positions for features
    bar_width = 0.8 / n_variables  # Width of each bar, distributed evenly


    # Create the plot
    plt.rcParams.update({'font.size': 16})

    fig, ax = plt.subplots(figsize=(15, 8))

    bar_containers = []  # collect per-variable bar groups

    # Plot each variable's importances
    for i, importances in enumerate(all_importances):
        bc = ax.bar(
            x + i * bar_width,  # Shift each group by `i * bar_width`
            importances,
            bar_width,
            label=variable_names[i]
        )
        bar_containers.append(bc)

    # Add labels and legend
    # Existing code (keep)
    ax.set_xlabel("Features")
    ax.set_ylabel("Importance")
    ax.set_title("Feature Importances Across Variables")
    ax.set_xticks(x + bar_width * (n_variables - 1) / 2)
    ax.set_xticklabels(features, rotation=25, ha='right', fontsize=10)

    # full variable labels (unsorted, plotting order)
    colors = [bc.patches[0].get_facecolor() for bc in bar_containers]

    # --- alphabetical legend prep (unchanged except `colors` already defined earlier) ---
    sorted_pairs = sorted(zip(variable_names, colors), key=lambda t: t[0].lower())
    variables_s, colors_s = zip(*sorted_pairs)

    # escape underscores if usetex
    if mpl.rcParams.get("text.usetex", False):
        legend_labels = [v.replace('_', r'\_') for v in variables_s]
    else:
        legend_labels = list(variables_s)

    legend_handles = [mpatches.Patch(color=c, label=lbl)
                      for c, lbl in zip(colors_s, legend_labels)]

    max_rows = 4
    ncol = int(np.ceil(len(legend_handles) / max_rows))

    fig = ax.figure

    # --- SPACING KNOBS ------------------------------------------------------
    XTICK_PAD = 2  # pixels between axis line and tick labels (smaller brings ticks up)
    XLABEL_PAD = 18  # points between tick labels and axis label (bigger pushes label down)
    LEGEND_PAD = 0.10  # figure fraction *below* axes: 0.10 ~= 10% of fig height
    BOTTOM_PAD = 0.32 + LEGEND_PAD  # final space reserved at bottom
    # ------------------------------------------------------------------------

    # 1) tighten tick labels up toward plot (so they don't collide with legend)
    ax.tick_params(axis='x', which='major', pad=XTICK_PAD)

    # 2) push the axis label a bit further below tick labels
    ax.set_xlabel("Features", labelpad=XLABEL_PAD)

    # 3) put legend farther below the axes using a negative y anchor
    #    y = -LEGEND_PAD means "LEGEND_PAD * axes height below the axes box"
    fig.legend(legend_handles,
               [h.get_label() for h in legend_handles],
               title="Variables",
               loc="upper center",
               bbox_to_anchor=(0.5, -LEGEND_PAD),  # << move down
               ncol=ncol,
               fontsize=7,
               title_fontsize=8,
               frameon=False,
               columnspacing=0.8,
               handlelength=1.0,
               handletextpad=0.4)

    # 4) reserve enough bottom margin so the legend & ticks fit
    fig.subplots_adjust(bottom=BOTTOM_PAD)

    # Save
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.25)
    plt.close()
    print(f"Feature importance comparison plot saved to {filename}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate F1 score and feature importance plots from multiple configuration JSON files.")
    parser.add_argument('-n', '--num_configs', type=int, default=47,
                        help="Number of configuration files to process (default: 17)")
    parser.add_argument('-p', '--config_prefix', type=str, default='rotated_config_',
                        help="Prefix for the configuration files (default: rotated_config_)")
    parser.add_argument('-s', '--suffix', type=str, default='.json',
                        help="Suffix for the configuration files (default: .json)")

    # Parse the arguments
    args = parser.parse_args()

    # Store results for plotting
    var_list = []
    # f1_weighted_cnn = []
    f1_macro_cnn = []
    f1_weighted_rf = []
    f1_macro_rf = []
    all_importances = []  # Collect all feature importances
    feature_names = None  # To store feature names from the first config

    # Iterate over all configuration files
    for i in range(1, args.num_configs + 1):
        config_file = f"{args.config_prefix}{i}{args.suffix}"
        if os.path.exists(config_file):
            if i not in [38, 39]:
                print(f"Processing {config_file}...")
                (variable,
                 weighted_rf, macro_rf, feature_importances, rf_feature_list) = process_config(config_file)

                var_list.append(variable)
                f1_weighted_rf.append(weighted_rf)
                f1_macro_rf.append(macro_rf)

                # Save feature importances and feature names
                if feature_importances is not None:
                    all_importances.append(feature_importances)
                    if feature_names is None:  # Store feature names once
                        feature_names = rf_feature_list

        else:
            print(f"Configuration file {config_file} not found. Skipping.")

    # Plot Weighted F1 Scores
    plot_f1_scores(var_list, f1_weighted_rf, "Weighted", "data/f1_score_comparison_weighted.png")

    # Plot Macro F1 Scores
    plot_f1_scores(var_list, f1_macro_rf, "Macro", "data/f1_score_comparison_macro.png")

    # Plot Feature Importances for all variables
    if all_importances and feature_names:
        plot_feature_importances(
            feature_names,
            all_importances,
            var_list,
            "data/feature_importances_rf_comparison.png"
        )


if __name__ == "__main__":
    main()