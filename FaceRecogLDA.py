import numpy as np
import os
from skimage import io
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Dataset parameters
dataset = "C:/Users/ahmed/OneDrive/Desktop/FaceRegocnition/ORL" 
num_subjects = 40
images_per_subject = 10
image_size = (92, 112)

def load_data(data_path, num_subjects, images_per_subject, image_size):
    data = []
    labels = []
    for subject_id in range(1, num_subjects + 1):
        subject_folder = os.path.join(data_path, f"s{subject_id}")
        for img_idx in range(1, images_per_subject + 1):
            img_path = os.path.join(subject_folder, f"{img_idx}.pgm")
            img = io.imread(img_path)
            img_data = img.flatten()  # Flatten image
            data.append(img_data)
            labels.append(subject_id)
    return np.array(data), np.array(labels)

# Load data
data, labels = load_data(dataset, num_subjects, images_per_subject, image_size)

# Separate training and test data based on odd/even rows
training_data = data[::2]
test_data = data[1::2]

# Extract labels
training_labels = labels[::2]
test_labels = labels[1::2]


# Perform LDA
lda = LinearDiscriminantAnalysis()
lda.fit(training_data, training_labels)

# Project training and test data onto the LDA subspace
projected_training_data = lda.transform(training_data)
projected_test_data = lda.transform(test_data)

# K-NN Classification with different K values
k_values = [1, 3, 5, 7]
accuracies = []
for k in k_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(projected_training_data, training_labels)
    predictions = knn_classifier.predict(projected_test_data)
    accuracy = accuracy_score(test_labels, predictions)
    accuracies.append(accuracy)

# Print accuracy results
print("K-NN Classification Accuracies with LDA ")
for k, accuracy in zip(k_values, accuracies):
    print(f"K = {k}: Accuracy = {accuracy:.4f}")

