import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate loading a confusion matrix
# Replace this with the actual loading code when you have the file
confusion_matrix = np.load('./data2/confusion_matrix_dssim_10_RF_local_dssim1rf.npy')

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(4, 4))
labels = ['zfp1e-3']
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels)  # Set cbar=False to remove the color bar
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('./data/confusion_matrix_heatmap.png')  # Save the heatmap as a .png file
plt.show()

# Print the values of the confusion matrix to a .txt file
with open('./data/confusion_matrix_values.txt', 'w') as f:
    for row in confusion_matrix:
        f.write(' '.join(map(str, row)) + '\n')