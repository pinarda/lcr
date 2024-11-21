import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import argparse
import json
from collections import Counter
import os

def process_config(config_file):
    """Load data and generate plots for a single configuration."""
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Extract parameters
    storageloc = config.get('StorageLoc')           # "./data/"
    times = config.get('Times')                     # [60]
    time = times[0]
    j = 0
    jobid = 0
    var_list = config.get('VarList')                # ["TS", "PRECT"]
    modeltype = config.get('ModelType')             # "cnn"

    # Load data for CNN
    test_labels_np_cnn = np.load(f"{storageloc}/test_labels_{j}{time}cnn{jobid}_{var_list[0]}.npy")
    test_predictions_cnn = np.load(f"{storageloc}/test_predictions_{j}{time}cnn{jobid}_{var_list[0]}.npy")
    label_encoder_cnn = np.load(f"{storageloc}/label_encoder_{j}{time}cnn{jobid}_{var_list[0]}.pkl", allow_pickle=True)

    # Load data for RF
    test_labels_np_rf = np.load(f"{storageloc}/test_labels_{j}{time}rf{jobid}_{var_list[0]}.npy")
    test_predictions_rf = np.load(f"{storageloc}/test_predictions_{j}{time}rf{jobid}_{var_list[0]}.npy")
    label_encoder_rf = np.load(f"{storageloc}/label_encoder_{j}{time}rf{jobid}_{var_list[0]}.pkl", allow_pickle=True)

    # Get all possible class labels from both encoders
    all_possible_labels = sorted(set(label_encoder_cnn.classes_).union(label_encoder_rf.classes_))

    # Map the string labels back to numeric indices for counting
    numeric_labels_cnn = label_encoder_cnn.transform(all_possible_labels)
    numeric_labels_rf = label_encoder_rf.transform(all_possible_labels)

    # Count occurrences of each numeric label in true labels, CNN predictions, and RF predictions
    true_label_counts = Counter(test_labels_np_cnn)
    cnn_prediction_counts = Counter(test_predictions_cnn)
    rf_prediction_counts = Counter(test_predictions_rf)

    # Ensure counts are in the same order for all
    true_counts = [true_label_counts.get(label, 0) for label in numeric_labels_cnn]
    cnn_counts = [cnn_prediction_counts.get(label, 0) for label in numeric_labels_cnn]
    rf_counts = [rf_prediction_counts.get(label, 0) for label in numeric_labels_rf]

    # Set up the bar positions
    x = np.arange(len(all_possible_labels))  # Label positions
    bar_width = 0.3  # Width of each bar

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # True label bars
    ax.bar(x - bar_width, true_counts, bar_width, label="True Labels", color="black", edgecolor="black")

    # CNN prediction bars
    ax.bar(x, cnn_counts, bar_width, label="CNN Predictions", color="blue", edgecolor="black")

    # RF prediction bars
    ax.bar(x + bar_width, rf_counts, bar_width, label="RF Predictions", color="red", edgecolor="black")

    # Adding labels and legend
    ax.set_xlabel("Labels")
    ax.set_ylabel("Counts")
    ax.set_title(f"True Labels vs. CNN Predictions vs. RF Predictions ({var_list[0]})")
    ax.set_xticks(x)
    ax.set_xticklabels(all_possible_labels, rotation=45, ha='right')  # Rotate for better readability
    ax.legend()

    # Save the plot
    plot_filename = f"{storageloc}/comparison_bar_plot_{j}{time}{modeltype}{jobid}_{var_list[0]}.png"
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()
    print(f"Plot saved to {plot_filename}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate plots from multiple configuration JSON files.")
    parser.add_argument('-n', '--num_configs', type=int, default=17,
                        help="Number of configuration files to process (default: 17)")
    parser.add_argument('-p', '--config_prefix', type=str, default='rotated_config_',
                        help="Prefix for the configuration files (default: rotated_config_)")
    parser.add_argument('-s', '--suffix', type=str, default='.json',
                        help="Suffix for the configuration files (default: .json)")

    # Parse the arguments
    args = parser.parse_args()

    # Iterate over all configuration files
    for i in range(1, args.num_configs + 1):
        config_file = f"{args.config_prefix}{i}{args.suffix}"
        if os.path.exists(config_file):
            print(f"Processing {config_file}...")
            process_config(config_file)
        else:
            print(f"Configuration file {config_file} not found. Skipping.")


if __name__ == "__main__":
    main()
