import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import argparse
import json
from collections import Counter


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process a configuration JSON file.")
    parser.add_argument('-c', '--config', type=str, default='rotated_config_1.json',
                        help="Path to the configuration JSON file (default: config.json)")

    # Parse the arguments
    args = parser.parse_args()

    # Read the specified JSON configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Your existing code that processes 'config' goes here
    print("Loaded configuration:", config)

    # Extract parameters
    base_orig_path = config.get('OrigPath')          # e.g., "/Users/alex/git/ldcpy/data/cam-fv/orig/"
    base_comp_path = config.get('CompPath')          # e.g., "/Users/alex/git/ldcpy/data/cam-fv/"
    var_list = config.get('VarList')                 # ["TS", "PRECT"]
    filename_pre = config.get('FilenamePre')         # ["", ""]
    filename_post = config.get('FilenamePost')       # [".100days.nc", ".100days.nc"]
    sub_dirs = config.get('SubDirs')                 # ["ens1", "ens2"]
    comp_dirs = config.get('CompDirs')               # ["zfp_1.0", "zfp_1e-1", "zfp_1e-3"]
    opt_ldcpy_dev_path = config.get('OptLdcpyDevPath')  # "/Users/alex/git/ldcpy"
    times = config.get('Times')                      # [60]
    navg = config.get('Navg')                        # 1
    storageloc = config.get('StorageLoc')           # "./data/"
    stride = config.get('Stride')                    # 1
    cut_dataset = config.get('CutDataset')            # 0
    metric = config.get('Metric')                    # ["dssim", "pcc", "spre"]
    save_dir = config.get('SaveDir')                  # "/Users/alex/git/lcr/lcr/data_analysis/RFnew/plots/"
    modeltype = config.get('ModelType')               # "cnn"
    featurelist = config.get('RFFeatureList')           # ["mean", "ns_con_var"]
    flat_var_list =[var for group in var_list for var in group]

    time = times[0]
    j = 0
    jobid = 0

    print("This is the main function of the mainplots module.")

    train_data_np_cnn = np.load(f"{storageloc}/train_data_{j}{time}cnn{jobid}.npy")
    val_data_np_cnn = np.load(f"{storageloc}/val_data_{j}{time}cnn{jobid}.npy")
    test_data_np_cnn = np.load(f"{storageloc}/test_data_{j}{time}cnn{jobid}.npy")
    train_labels_np_cnn = np.load(f"{storageloc}/train_labels_{j}{time}cnn{jobid}.npy")
    val_labels_np_cnn = np.load(f"{storageloc}/val_labels_{j}{time}cnn{jobid}.npy")
    test_labels_np_cnn = np.load(f"{storageloc}/test_labels_{j}{time}cnn{jobid}_{var_list[0]}.npy")
    label_encoder_cnn = np.load(f"{storageloc}/label_encoder_{j}{time}cnn{jobid}_{var_list[0]}.pkl", allow_pickle=True)
    predictions_cnn = np.load(f"{storageloc}/predictions_{j}{time}cnn{jobid}_{var_list[0]}.npy")
    test_predictions_cnn = np.load(f"{storageloc}/test_predictions_{j}{time}cnn{jobid}_{var_list[0]}.npy")

    train_data_np_rf = np.load(f"{storageloc}/train_data_{j}{time}rf{jobid}.npy")
    val_data_np_rf = np.load(f"{storageloc}/val_data_{j}{time}rf{jobid}.npy")
    test_data_np_rf = np.load(f"{storageloc}/test_data_{j}{time}rf{jobid}.npy")
    train_labels_np_rf = np.load(f"{storageloc}/train_labels_{j}{time}rf{jobid}.npy")
    val_labels_np_rf = np.load(f"{storageloc}/val_labels_{j}{time}rf{jobid}.npy")
    test_labels_np_rf = np.load(f"{storageloc}/test_labels_{j}{time}rf{jobid}_{var_list[0]}.npy")
    label_encoder_rf = np.load(f"{storageloc}/label_encoder_{j}{time}rf{jobid}_{var_list[0]}.pkl", allow_pickle=True)
    predictions_rf = np.load(f"{storageloc}/predictions_{j}{time}rf{jobid}_{var_list[0]}.npy")
    test_predictions_rf = np.load(f"{storageloc}/test_predictions_{j}{time}rf{jobid}_{var_list[0]}.npy")

    # Compute accuracy for CNN
    accuracy_cnn = accuracy_score(test_labels_np_cnn, test_predictions_cnn)
    print(f"Test Accuracy: {accuracy_cnn:.4f}")

    # Compute confusion matrix
    conf_matrix_cnn = confusion_matrix(test_labels_np_cnn, test_predictions_cnn)
    print("Confusion Matrix:")
    print(conf_matrix_cnn)

    # Compute classification report
    class_report_cnn = classification_report(test_labels_np_cnn, test_predictions_cnn)
    print("Classification Report:")
    print(class_report_cnn)

    # Compute accuracy for RF
    accuracy_rf = accuracy_score(test_labels_np_rf, test_predictions_rf)
    print(f"Test Accuracy: {accuracy_rf:.4f}")

    # Compute confusion matrix
    conf_matrix_rf = confusion_matrix(test_labels_np_rf, test_predictions_rf)
    print("Confusion Matrix:")
    print(conf_matrix_rf)

    # Compute classification report
    class_report_rf = classification_report(test_labels_np_rf, test_predictions_rf)
    print("Classification Report:")
    print(class_report_rf)

    all_labels_cnn = list(label_encoder_cnn.classes_)
    all_labels_rf = list(label_encoder_rf.classes_)

    # Calculate the column sums
    col_sums_cnn = conf_matrix_cnn.sum(axis=0)
    col_sums_rf = conf_matrix_rf.sum(axis=0)

    # Extract diagonal (correct predictions) and off-diagonal values for plotting
    diag_cnn = np.diag(conf_matrix_cnn)
    off_diag_cnn = col_sums_cnn - diag_cnn
    diag_rf = np.diag(conf_matrix_rf)
    off_diag_rf = col_sums_rf - diag_rf

    # Plotting
    fig, ax = plt.subplots()

    # Plot correct predictions in bold color
    ax.bar(all_labels_cnn, diag_cnn, label="Correct Predictions", color="blue", edgecolor="black")

    # Plot misclassifications in a lighter color, stacked on top of correct predictions
    ax.bar(all_labels_cnn, off_diag_cnn, bottom=diag_cnn, label="Misclassifications", color="lightblue", edgecolor="black")

    # Adding labels and legend
    ax.set_xlabel("Labels")
    ax.set_ylabel("Counts")
    ax.set_title("Confusion Matrix Summary")
    ax.legend()

    plt.show()
    # save the plot
    # fig.savefig(f"{storageloc}/confusion_matrix_plot_{j}{time}{modeltype}{jobid}_{var_list[0]}.png")


    # Count occurrences of each class in true labels, CNN predictions, and RF predictions
    true_label_counts = Counter(test_labels_np_cnn)
    cnn_prediction_counts = Counter(test_predictions_cnn)
    rf_prediction_counts = Counter(test_predictions_rf)

    # Get all unique labels across true labels, CNN predictions, and RF predictions
    all_labels = sorted(set(true_label_counts.keys()).union(cnn_prediction_counts.keys(), rf_prediction_counts.keys()))

    # Ensure counts are in the same order for all
    true_counts = [true_label_counts.get(label, 0) for label in all_labels]
    cnn_counts = [cnn_prediction_counts.get(label, 0) for label in all_labels]
    rf_counts = [rf_prediction_counts.get(label, 0) for label in all_labels]

    # Set up the bar positions
    x = np.arange(len(all_labels))  # Label positions
    bar_width = 0.3  # Width of each bar

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # True label bars
    ax.bar(x - bar_width, true_counts, bar_width, label="True Labels", color="green", edgecolor="black")

    # CNN prediction bars
    ax.bar(x, cnn_counts, bar_width, label="CNN Predictions", color="blue", edgecolor="black")

    # RF prediction bars
    ax.bar(x + bar_width, rf_counts, bar_width, label="RF Predictions", color="orange", edgecolor="black")

    # Adding labels and legend
    ax.set_xlabel("Labels")
    ax.set_ylabel("Counts")
    ax.set_title("True Labels vs. CNN Predictions vs. RF Predictions")
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels)
    ax.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()

    # Save the plot
    fig.savefig(f"{storageloc}/comparison_bar_plot_{j}{time}{modeltype}{jobid}_{var_list[0]}.png")


if __name__ == "__main__":
    main()