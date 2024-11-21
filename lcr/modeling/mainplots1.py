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
    var_list = config.get('VarList')                # ["TS", "PRECT"]

    # Load data for CNN
    test_labels_np_cnn = np.load(f"{storageloc}/test_labels_{j}{time}cnn{jobid}_{var_list[0]}.npy")
    test_predictions_cnn = np.load(f"{storageloc}/test_predictions_{j}{time}cnn{jobid}_{var_list[0]}.npy")

    # Load data for RF
    test_labels_np_rf = np.load(f"{storageloc}/test_labels_{j}{time}rf{jobid}_{var_list[0]}.npy")
    test_predictions_rf = np.load(f"{storageloc}/test_predictions_{j}{time}rf{jobid}_{var_list[0]}.npy")

    # Compute F1 scores
    f1_cnn = f1_score(test_labels_np_cnn, test_predictions_cnn, average='weighted')
    f1_rf = f1_score(test_labels_np_rf, test_predictions_rf, average='weighted')

    return var_list[0], f1_cnn, f1_rf


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate F1 score plot from multiple configuration JSON files.")
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
    f1_scores_cnn = []
    f1_scores_rf = []

    # Iterate over all configuration files
    for i in range(1, args.num_configs + 1):
        config_file = f"{args.config_prefix}{i}{args.suffix}"
        if os.path.exists(config_file):
            print(f"Processing {config_file}...")
            var, f1_cnn, f1_rf = process_config(config_file)
            var_list.append(var)
            f1_scores_cnn.append(f1_cnn)
            f1_scores_rf.append(f1_rf)
        else:
            print(f"Configuration file {config_file} not found. Skipping.")

    # Plotting
    x = np.arange(len(var_list))  # X-axis positions
    bar_width = 0.4  # Width of each bar

    fig, ax = plt.subplots(figsize=(10, 6))

    # CNN F1 scores
    ax.bar(x - bar_width / 2, f1_scores_rf, bar_width, label="CNN F1 Score", color="blue", edgecolor="black")

    # RF F1 scores
    ax.bar(x + bar_width / 2, f1_scores_rf , bar_width, label="RF F1 Score", color="red", edgecolor="black")

    # Adding labels and legend
    ax.set_xlabel("Variables")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score Comparison for CNN and RF Models")
    ax.set_xticks(x)
    ax.set_xticklabels(var_list, rotation=45, ha='right')  # Rotate for better readability
    ax.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()

    # Save the plot
    plot_filename = f"data/f1_score_comparison.png"
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")


if __name__ == "__main__":
    main()
