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

    # Flatten var_list and take the first element of the first list
    variable = var_list[0][0] if var_list and isinstance(var_list[0], list) else "Unknown"

    # Load data for CNN
    test_labels_np_cnn = np.load(f"{storageloc}/test_labels_{j}{time}cnn{jobid}_{variable}.npy")
    test_predictions_cnn = np.load(f"{storageloc}/test_predictions_{j}{time}cnn{jobid}_{variable}.npy")

    # Load data for RF
    test_labels_np_rf = np.load(f"{storageloc}/test_labels_{j}{time}rf{jobid}_{variable}.npy")
    test_predictions_rf = np.load(f"{storageloc}/test_predictions_{j}{time}rf{jobid}_{variable}.npy")

    # Compute F1 scores
    f1_weighted_cnn = f1_score(test_labels_np_cnn, test_predictions_cnn, average='weighted')
    f1_macro_cnn = f1_score(test_labels_np_cnn, test_predictions_cnn, average='macro')
    f1_weighted_rf = f1_score(test_labels_np_rf, test_predictions_rf, average='weighted')
    f1_macro_rf = f1_score(test_labels_np_rf, test_predictions_rf, average='macro')

    return variable, f1_weighted_cnn, f1_macro_cnn, f1_weighted_rf, f1_macro_rf


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate F1 score plots from multiple configuration JSON files.")
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

    # Plotting Weighted F1 Scores
    x = np.arange(len(var_list))  # X-axis positions
    bar_width = 0.2  # Width of each bar

    fig, ax = plt.subplots(figsize=(12, 8))

    # CNN Weighted F1 scores
    ax.bar(x - bar_width * 1.5, f1_weighted_rf, bar_width, label="CNN Weighted F1", color="blue", edgecolor="black")

    # CNN Macro F1 scores
    ax.bar(x - bar_width / 2, f1_macro_rf, bar_width, label="CNN Macro F1", color="lightblue", edgecolor="black")

    # RF Weighted F1 scores
    ax.bar(x + bar_width / 2, f1_weighted_rf, bar_width, label="RF Weighted F1", color="orange", edgecolor="black")

    # RF Macro F1 scores
    ax.bar(x + bar_width * 1.5, f1_macro_rf, bar_width, label="RF Macro F1", color="gold", edgecolor="black")

    # Adding labels and legend
    ax.set_xlabel("Variables")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score Comparison for CNN and RF Models (Weighted and Macro)")
    ax.set_xticks(x)
    ax.set_xticklabels(var_list, rotation=45, ha='right')  # Rotate for better readability
    ax.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()

    # Save the plot
    plot_filename = f"f1_score_comparison_weighted_macro.png"
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")


if __name__ == "__main__":
    main()
