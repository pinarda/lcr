import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import argparse
import json
import os

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

    # Use var_list[0] for file names directly
    variable_file = var_list[0]
    # Use the actual string for display purposes
    variable_display = var_list[0][0] if isinstance(var_list[0], list) and var_list[0] else "Unknown"

    # Load data for CNN
    test_labels_np_cnn = np.load(f"{storageloc}/test_labels_{j}{time}cnn{jobid}_{variable_file}.npy")
    test_predictions_cnn = np.load(f"{storageloc}/test_predictions_{j}{time}cnn{jobid}_{variable_file}.npy")

    # Load data for RF
    test_labels_np_rf = np.load(f"{storageloc}/test_labels_{j}{time}rf{jobid}_{variable_file}.npy")
    test_predictions_rf = np.load(f"{storageloc}/test_predictions_{j}{time}rf{jobid}_{variable_file}.npy")

    # Compute F1 scores
    f1_weighted_cnn = f1_score(test_labels_np_cnn, test_predictions_cnn, average='weighted')
    f1_macro_cnn = f1_score(test_labels_np_cnn, test_predictions_cnn, average='macro')
    f1_weighted_rf = f1_score(test_labels_np_rf, test_predictions_rf, average='weighted')
    f1_macro_rf = f1_score(test_labels_np_rf, test_predictions_rf, average='macro')

    return variable_display, f1_weighted_cnn, f1_macro_cnn, f1_weighted_rf, f1_macro_rf


def plot_f1_scores(var_list, cnn_scores, rf_scores, metric_name, filename):
    """Create a bar plot for F1 scores."""
    x = np.arange(len(var_list))  # X-axis positions
    bar_width = 0.35  # Width of each bar

    fig, ax = plt.subplots(figsize=(10, 6))

    # CNN F1 scores
    ax.bar(x - bar_width / 2, rf_scores, bar_width, label="CNN F1 Score", color="blue", edgecolor="black")

    # RF F1 scores
    ax.bar(x + bar_width / 2, rf_scores, bar_width, label="RF F1 Score", color="red", edgecolor="black")

    # Adding labels and legend
    ax.set_xlabel("Variables")
    ax.set_ylabel(f"{metric_name} F1 Score")
    ax.set_title(f"{metric_name} F1 Score Comparison for CNN and RF Models")
    ax.set_xticks(x)
    ax.set_xticklabels(var_list, rotation=45, ha='right')  # Rotate for better readability
    ax.legend()

    # Display and save the plot
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"{metric_name} F1 score plot saved to {filename}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate separate F1 score plots from multiple configuration JSON files.")
    parser.add_argument('-n', '--num_configs', type=int, default=17,
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

    # Iterate over all configuration files
    for i in range(1, args.num_configs + 1):
        config_file = f"{args.config_prefix}{i}{args.suffix}"
        if os.path.exists(config_file):
            print(f"Processing {config_file}...")
            (variable, weighted_cnn, macro_cnn,
             weighted_rf, macro_rf) = process_config(config_file)
            var_list.append(variable)
            f1_weighted_cnn.append(weighted_cnn)
            f1_macro_cnn.append(macro_cnn)
            f1_weighted_rf.append(weighted_rf)
            f1_macro_rf.append(macro_rf)
        else:
            print(f"Configuration file {config_file} not found. Skipping.")

    # Plot Weighted F1 Scores
    plot_f1_scores(var_list, f1_weighted_cnn, f1_weighted_rf, "Weighted", "data/f1_score_comparison_weighted.png")

    # Plot Macro F1 Scores
    plot_f1_scores(var_list, f1_macro_cnn, f1_macro_rf, "Macro", "data/f1_score_comparison_macro.png")


if __name__ == "__main__":
    main()
