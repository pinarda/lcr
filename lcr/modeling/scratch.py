import glob
import numpy as np

def get_variable_name(file_name):
    """
    Extracts the variable name from a file name.
    """
    return file_name.split("_")[-1].strip(".npy").strip("[]").strip("'")

def count_labels(data):
    """
    Counts the occurrences of 0 and 1 in the data.
    """
    count_0 = np.sum(data == 0)
    count_1 = np.sum(data == 1)
    return count_0, count_1

def print_combined_counts_and_generate_latex():
    """
    Reads .npy files for test labels and predictions, counts occurrences of 0 and 1,
    and intelligently matches files based on variable names.
    Outputs counts and generates a LaTeX table.
    """
    label_file_pattern = '/Users/alex/Casper/1211/labels/test_labels*.npy'
    prediction_file_pattern = '/Users/alex/Casper/1211/labels/test_predictions*.npy'

    label_files = glob.glob(label_file_pattern)
    prediction_files = glob.glob(prediction_file_pattern)

    if not label_files:
        print("No matching label files found.")
    if not prediction_files:
        print("No matching prediction files found.")

    # Create dictionaries to map variable names to file paths
    label_files_dict = {get_variable_name(f): f for f in label_files}
    prediction_files_dict = {get_variable_name(f): f for f in prediction_files}

    # Get all unique variable names from both labels and predictions
    all_variables = set(label_files_dict.keys()).union(prediction_files_dict.keys())

    # Start building the LaTeX table
    latex_table = []
    latex_table.append("\\begin{table*}[h!]")
    latex_table.append("\\centering")
    latex_table.append("\\begin{tabular}{|c|c|c|}")
    latex_table.append("\\hline")
    latex_table.append("\\rowcolor{gray!25} \\textbf{Variable} & \\textbf{zfp\\_p\\_16 labels/predictions} & \\textbf{zfp\\_p\\_26 labels/predictions}\\\\ \\hline")

    i = 1

    for variable in sorted(all_variables):
        label_file = label_files_dict.get(variable)
        prediction_file = prediction_files_dict.get(variable)

        label_counts = (0, 0)  # Default counts if no label file
        prediction_counts = (0, 0)  # Default counts if no prediction file

        if label_file:
            try:
                labels_data = np.load(label_file)
                label_counts = count_labels(labels_data)
            except Exception as e:
                print(f"Failed to process label file {label_file}: {e}")

        if prediction_file:
            try:
                predictions_data = np.load(prediction_file)
                prediction_counts = count_labels(predictions_data)
            except Exception as e:
                print(f"Failed to process prediction file {prediction_file}: {e}")

        # Print combined counts for the variable
        print(f"{variable}: "
              f"Labels - zfp_p_16, {label_counts[0]}, zfp_p_26, {label_counts[1]}; "
              f"Predictions - zfp_p_16, {prediction_counts[0]}, zfp_p_26, {prediction_counts[1]}")

        # Add row to the LaTeX table with alternating gray rows
        # Add row to the LaTeX table with alternating gray rows
        if i % 2 == 0:
            latex_table.append(f"\\rowcolor{{gray!20}}")
        latex_table.append(f"{variable} & {label_counts[0]}/{prediction_counts[0]} & {label_counts[1]}/{prediction_counts[1]} \\\\ \\hline")
        i += 1

    # End the LaTeX table
    latex_table.append("\\end{tabular}")
    latex_table.append("\\caption{Counts of Labels and Predictions}")
    latex_table.append("\\label{tab:counts}")
    latex_table.append("\\end{table*}")

    # Print the LaTeX table
    print("\n".join(latex_table))

if __name__ == "__main__":
    print_combined_counts_and_generate_latex()