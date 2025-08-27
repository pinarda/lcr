import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import argparse
import json
import os
import pandas as pd

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

    test_labels_np_cnn = np.load(f"{storageloc}/test_labels_02000cnn0_{variable_file}.npy")
    test_predictions_cnn = np.load(f"{storageloc}/test_predictions_02000cnn0_{variable_file}.npy")

    # Load data for RF
    test_labels_np_rf = load_first_existing(
        [
            f"{storageloc}/test_labels_02000rf0_{variable_file}.npy",
            f"{storageloc}/test_labels_12000rf0_{variable_file}.npy",
            f"{storageloc}/test_labels_22000rf0_{variable_file}.npy",
            f"{storageloc}/test_labels_32000rf0_{variable_file}.npy",
            f"{storageloc}/test_labels_42000rf0_{variable_file}.npy",
        ],
        allow_pickle=True
    )

        # np.load(f"{storageloc}/test_labels_02000rf0_{variable_file}.npy"))
    # test_predictions_rf = np.load(f"{storageloc}/test_predictions_rf_{j}{time}rf{jobid}_{variable_file}.npy")
    test_predictions_rf = load_first_existing(
        [
            f"{storageloc}/test_predictions_rf_02000rf0_{variable_file}.npy",
            f"{storageloc}/test_predictions_rf_12000rf0_{variable_file}.npy",
            f"{storageloc}/test_predictions_rf_22000rf0_{variable_file}.npy",
            f"{storageloc}/test_predictions_rf_32000rf0_{variable_file}.npy",
            f"{storageloc}/test_predictions_rf_42000rf0_{variable_file}.npy",
        ],
        allow_pickle=True
    )

    # Compute F1 scores
    f1_weighted_cnn = f1_score(test_labels_np_cnn, test_predictions_cnn, average='weighted')
    f1_macro_cnn = f1_score(test_labels_np_cnn, test_predictions_cnn, average='macro')
    f1_weighted_rf = f1_score(test_labels_np_rf, test_predictions_rf, average='weighted')
    f1_macro_rf = f1_score(test_labels_np_rf, test_predictions_rf, average='macro')

    print(f"CNN: {f1_macro_cnn} RF: {f1_macro_rf},  F1 score plot for {var_list[0]}")

    # Load feature importances for RF
    feature_importance_file = f"{storageloc}/feature_importances_rf_02000rf0_{variable_file}.npy"
    if os.path.exists(feature_importance_file):
        feature_importances = np.load(feature_importance_file)
    else:
        feature_importances = None

    # Create a table with counts
    compression_labels = {0: "zfp_p_8", 1: "zfp_p_10", 2: "zfp_p_12", 3: "zfp_p_14", 4: "zfp_p_18", 5: "zfp_p_20", 6: "zfp_p_22", 7: "zfp_p_24"}
    table_data = []

    for label, prediction in zip(test_labels_np_rf, test_predictions_rf):
        label_name = compression_labels.get(label, "Unknown")
        prediction_name = compression_labels.get(prediction, "Unknown")
        table_data.append((variable_file[0], label_name, prediction_name))

    # Convert to DataFrame
    df = pd.DataFrame(table_data, columns=["Variable", "Compression Label", "Prediction Label"])

    # Count occurrences of each Label-Prediction Pair
    df_counts = df.groupby(["Variable", "Compression Label", "Prediction Label"]).size().reset_index(name="Count")

    return variable_display, f1_weighted_cnn, f1_macro_cnn, f1_weighted_rf, f1_macro_rf, feature_importances, rf_feature_list, df_counts


def plot_f1_scores(var_list, cnn_scores, rf_scores, metric_name, filename):
    """Create a bar plot for F1 scores."""
    x = np.arange(len(var_list))  # X-axis positions
    bar_width = 0.35  # Width of each bar

    ones = np.ones(len(var_list))
    fig, ax = plt.subplots(figsize=(10, 6))

    # CNN F1 scores
    ax.bar(x - bar_width / 2, cnn_scores, bar_width, label="CNN F1 Score", color="#e18683", edgecolor="black")

    # RF F1 scores
    ax.bar(x + bar_width / 2, rf_scores, bar_width, label="RF F1 Score", color="#B6D7E4", edgecolor="black")

    # Adding labels and legend
    ax.set_xlabel("Variables")
    ax.set_ylabel(f"{metric_name} F1 Score")
    ax.set_title(f"{metric_name} F1 Score Comparison for CNN and RF Models")
    ax.set_xticks(x)
    ax.set_xticklabels(var_list, rotation=45, ha='right', fontsize=8)  # Rotate for better readability
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Display and save the plot
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"{metric_name} F1 score plot saved to {filename}")


def plot_f1_scores_alpha(var_list, cnn_scores, rf_scores, metric_name, filename):
    """Create a bar plot for F1 scores, with variables shown alphabetically."""
    import numpy as np
    import matplotlib.pyplot as plt

    # ── sort everything by variable name ──────────────────────────────
    # zip → sort by the variable string (case-insensitive) → unzip
    sorted_rows = sorted(zip(var_list, cnn_scores, rf_scores),
                         key=lambda t: t[0].lower())
    var_sorted, cnn_sorted, rf_sorted = map(list, zip(*sorted_rows))

    # ── plotting ─────────────────────────────────────────────────────
    x         = np.arange(len(var_sorted))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - bar_width/2, cnn_sorted, bar_width,
           label="CNN F1 Score", color="#e18683", edgecolor="black")
    ax.bar(x + bar_width/2, rf_sorted,  bar_width,
           label="RF F1 Score",  color="#B6D7E4", edgecolor="black")

    ax.set_xlabel("Variables")
    ax.set_ylabel(f"{metric_name} F1 Score")
    ax.set_title(f"{metric_name} F1 Score Comparison for CNN and RF Models")

    ax.set_xticks(x)
    ax.set_xticklabels(var_sorted, rotation=45, ha='right', fontsize=8)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
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

    # Number of features and variables
    n_features = len(features)
    n_variables = len(all_importances)

    # X-axis positions
    x = np.arange(n_features)  # Positions for features
    bar_width = 0.8 / n_variables  # Width of each bar, distributed evenly

    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot each variable's importances
    for i, importances in enumerate(all_importances):
        ax.bar(
            x + i * bar_width,  # Shift each group by `i * bar_width`
            importances,
            bar_width,
            label=variable_names[i]
        )

    # Add labels and legend
    ax.set_xlabel("Features")
    ax.set_ylabel("Importance")
    ax.set_title("Feature Importances Across Variables")
    ax.set_xticks(x + bar_width * (n_variables - 1) / 2)  # Center the group of bars
    ax.set_xticklabels(features, rotation=45, ha='right')  # Rotate labels for better readability
    ax.legend(title="Variables", loc='upper left', bbox_to_anchor=(1.05, 1))

    # Save the plot
    plt.tight_layout()
    plt.savefig(filename)
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
    f1_weighted_cnn = []
    f1_macro_cnn = []
    f1_weighted_rf = []
    f1_macro_rf = []
    all_importances = []  # Collect all feature importances
    feature_names = None  # To store feature names from the first config
    all_data = []

    # Iterate over all configuration files
    for i in range(1, args.num_configs + 1):
        config_file = f"{args.config_prefix}{i}{args.suffix}"
        if os.path.exists(config_file):
            # if i not in [23, 24, 36, 37, 40, 41, 43, 44]:
            # if i not in [1, 2, 4, 5, 12, 13, 16, 19, 20, 21, 22, 24, 26, 27, 37, 43, 44, 45, 46, 47]:
            if i not in [20, 25]:
                print(f"Processing {config_file}...")
                (variable, weighted_cnn, macro_cnn,
                 weighted_rf, macro_rf, feature_importances, rf_feature_list, df_counts) = process_config(config_file)
                all_data.append(df_counts)

                var_list.append(variable)
                f1_weighted_cnn.append(weighted_cnn)
                f1_macro_cnn.append(macro_cnn)
                f1_weighted_rf.append(weighted_rf)
                f1_macro_rf.append(macro_rf)

                # Save feature importances and feature names
                if feature_importances is not None:
                    all_importances.append(feature_importances)
                    if feature_names is None:  # Store feature names once
                        feature_names = rf_feature_list

        else:
            print(f"Configuration file {config_file} not found. Skipping.")

    # Combine all data into a single DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)

    # Pivot the combined DataFrame
    df_pivot = combined_df.pivot_table(
        index="Variable",
        columns="Compression Label",
        values="Count",
        aggfunc="sum"
    ).fillna(0).astype(int)

    # Print the combined pivot table
    print("Combined Pivot Table:")
    print(df_pivot)

    # Format the LaTeX table with alternating gray rows
    latex_table = r"\begin{table*}[h!]" + "\n"
    latex_table += r"\centering" + "\n"
    latex_table += r"\begin{tabular}{|c|c|c|}" + "\n"
    latex_table += r"\hline" + "\n"
    latex_table += r"\rowcolor{gray!25} \textbf{Variable} & \textbf{zfp\_p\_16 (labels)} & \textbf{zfp\_p\_22 (labels)}\\ \hline" + "\n"

    # Add rows with alternating gray background
    for i, (variable, row) in enumerate(df_pivot.iterrows()):
        row_color = r"\rowcolor{gray!20} " if i % 2 else ""
        latex_table += f"{row_color}{variable} & {row['zfp_p_10']} & {row['zfp_p_12']} \\\\ \\hline\n"

    latex_table += r"\end{tabular}" + "\n"
    latex_table += r"\caption{Counts of labels only (predictions removed).}" + "\n"
    latex_table += r"\label{tab:counts_counts}" + "\n"
    latex_table += r"\end{table*}" + "\n"

    # Save the LaTeX table to a file
    latex_file_path = "./table_output.tex"  # Adjust path as needed
    with open(latex_file_path, "w") as f:
        f.write(latex_table)


    print(f"LaTeX table saved to {latex_file_path}")

    # Plot Weighted F1 Scores
    plot_f1_scores_alpha(var_list, f1_weighted_cnn, f1_weighted_rf, "Weighted", "data/f1_score_comparison_weighted.png")

    # Plot Macro F1 Scores
    plot_f1_scores_alpha(var_list, f1_macro_cnn, f1_macro_rf, "Macro", "data/f1_score_comparison_macro.png")

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
