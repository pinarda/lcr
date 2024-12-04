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
    rf_feature_list = config.get('RFFeatureList')   # Feature names for RF

    # Use var_list[0] for file names directly
    variable_file = var_list[0]
    # Use the actual string for display purposes
    variable_display = var_list[0][0] if isinstance(var_list[0], list) and var_list[0] else "Unknown"

    # Load data for CNN
    test_labels_np_cnn = np.load(f"{storageloc}/test_labels_{j}{time}cnn{jobid}_{variable_file}.npy")
    test_predictions_cnn = np.load(f"{storageloc}/test_predictions_rf_{j}{time}cnn{jobid}_{variable_file}.npy")

    # Load data for RF
    test_labels_np_rf = np.load(f"{storageloc}/test_labels_{j}{time}rf{jobid}_{variable_file}.npy")
    test_predictions_rf = np.load(f"{storageloc}/test_predictions_rf_{j}{time}rf{jobid}_{variable_file}.npy")

    # Compute F1 scores
    f1_weighted_cnn = f1_score(test_labels_np_cnn, test_predictions_cnn, average='weighted')
    f1_macro_cnn = f1_score(test_labels_np_cnn, test_predictions_cnn, average='macro')
    f1_weighted_rf = f1_score(test_labels_np_rf, test_predictions_rf, average='weighted')
    f1_macro_rf = f1_score(test_labels_np_rf, test_predictions_rf, average='macro')

    # Load feature importances for RF
    feature_importance_file = f"{storageloc}/feature_importances_rf_11600rf0_{variable_file}.npy"
    if os.path.exists(feature_importance_file):
        feature_importances = np.load(feature_importance_file)
    else:
        feature_importances = None

    return variable_display, f1_weighted_cnn, f1_macro_cnn, f1_weighted_rf, f1_macro_rf, feature_importances, rf_feature_list


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
    ax.legend(title="Variables")

    # Save the plot
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Feature importance comparison plot saved to {filename}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate F1 score and feature importance plots from multiple configuration JSON files.")
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
    all_importances = []  # Collect all feature importances
    feature_names = None  # To store feature names from the first config

    # Iterate over all configuration files
    for i in range(1, args.num_configs + 1):
        config_file = f"{args.config_prefix}{i}{args.suffix}"
        if os.path.exists(config_file):
            print(f"Processing {config_file}...")
            (variable, weighted_cnn, macro_cnn,
             weighted_rf, macro_rf, feature_importances, rf_feature_list) = process_config(config_file)

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

    # Plot Weighted F1 Scores
    plot_f1_scores(var_list, f1_weighted_cnn, f1_weighted_rf, "Weighted", "data/f1_score_comparison_weighted.png")

    # Plot Macro F1 Scores
    plot_f1_scores(var_list, f1_macro_cnn, f1_macro_rf, "Macro", "data/f1_score_comparison_macro.png")

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
