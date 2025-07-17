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

    Parameters
    ----------
    features : list[str]
        Feature (x-axis) names (length = n_features).
    all_importances : list[list[float] or 2D array-like]
        Outer length = n_variables; each inner seq length must equal len(features).
    variable_names : list[str]
        Names of the variables corresponding to rows in all_importances.
    filename : str
        Path to save the plot (extension dictates format, e.g., .png, .pdf).
    """
    import math
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # ------------------------------------------------------------------
    # Basic validation
    # ------------------------------------------------------------------
    if not all_importances or not features:
        print("No data for feature importance plot.")
        return

    n_features = len(features)
    n_variables = len(all_importances)

    if len(variable_names) != n_variables:
        raise ValueError(
            f"variable_names length ({len(variable_names)}) "
            f"!= all_importances length ({n_variables})"
        )

    # ensure each importance row matches feature count
    for i, imp in enumerate(all_importances):
        if len(imp) != n_features:
            raise ValueError(
                f"Row {i} ('{variable_names[i]}') len={len(imp)} "
                f"!= n_features ({n_features})"
            )

    # ------------------------------------------------------------------
    # Sort variables alphabetically (case-insensitive) BEFORE plotting
    # ------------------------------------------------------------------
    order = sorted(range(n_variables), key=lambda i: variable_names[i].lower())
    variables_plot = [variable_names[i] for i in order]
    importances_plot = [all_importances[i] for i in order]

    # ------------------------------------------------------------------
    # X positions
    # ------------------------------------------------------------------
    x = np.arange(n_features)
    bar_width = 0.8 / n_variables  # 80% of unit width shared across groups

    # ------------------------------------------------------------------
    # Figure / axes
    # widen figure slightly for many features
    fig_width = max(8, min(24, 0.35 * n_features))  # heuristic
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.rcParams.update({'font.size': 16})

    # ------------------------------------------------------------------
    # Colors (qualitative colormap; repeat if > colormap capacity)
    # ------------------------------------------------------------------
    cmap = plt.get_cmap('tab20')
    colors_plot = [cmap(i % cmap.N) for i in range(n_variables)]

    # ------------------------------------------------------------------
    # Plot grouped bars (sorted order)
    # ------------------------------------------------------------------
    bar_containers = []
    for i, (var, imp) in enumerate(zip(variables_plot, importances_plot)):
        bc = ax.bar(
            x + i * bar_width,
            imp,
            bar_width,
            color=colors_plot[i],
            label="_nolegend_"  # suppress auto legend
        )
        bar_containers.append(bc)

    # ------------------------------------------------------------------
    # Axis labels & ticks
    # ------------------------------------------------------------------
    ax.set_ylabel("Importance", fontsize=14)
    ax.set_title("Feature Importances Across Variables")

    # x ticks centered on each feature group
    ax.set_xticks(x + bar_width * (n_variables - 1) / 2)

    # smaller font; rotate for space
    ax.set_xticklabels(features, rotation=25, ha='right', fontsize=14)

    # push x-axis label away from ticks (adjust below in spacing knobs)
    # we'll set label after spacing knobs so we can change labelpad
    # (Matplotlib doesn't overwrite labelpad if unchanged, but order is clear)

    # ------------------------------------------------------------------
    # Legend (variables_plot already sorted)
    # ------------------------------------------------------------------
    # escape underscores when usetex is on
    if mpl.rcParams.get("text.usetex", False):
        legend_labels = [v.replace('_', r'\_') for v in variables_plot]
    else:
        legend_labels = variables_plot

    legend_handles = [
        mpatches.Patch(color=c, label=lbl)
        for c, lbl in zip(colors_plot, legend_labels)
    ]

    # wrap legend across multiple columns (aim ≤ 4 rows)
    max_rows = 4
    ncol = int(math.ceil(len(legend_handles) / max_rows))

    # ------------------------------------------------------------------
    # Spacing knobs
    # ------------------------------------------------------------------
    XTICK_PAD   = 2     # px from axis spine to tick labels
    XLABEL_PAD  = 25    # pts from tick labels to axis label
    LEGEND_PAD  = 0.06  # fraction of axes height below axes
    # bottom margin: base for xticks + legend footprint
    BOTTOM_PAD  = 0.12 + LEGEND_PAD

    # Apply spacing tweaks
    ax.tick_params(axis='x', which='major', pad=XTICK_PAD)
    ax.set_xlabel("Features", labelpad=XLABEL_PAD, fontsize=14)

    # Figure-level legend so it spans full width
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        title="Variables",
        loc="upper center",
        bbox_to_anchor=(0.5, -LEGEND_PAD),  # move below axes
        ncol=ncol,
        fontsize=10,
        title_fontsize=12,
        frameon=False,
        columnspacing=0.8,
        handlelength=1.0,
        handletextpad=0.4,
    )

    # Reserve space at bottom for ticks + legend
    fig.subplots_adjust(bottom=BOTTOM_PAD)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)
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

    # Parse the argument
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