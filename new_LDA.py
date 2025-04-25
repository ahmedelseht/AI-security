import os
from skimage import io
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Dataset parameters
dataset_path = "C:/Users/ahmed/OneDrive/Desktop/FaceRegocnition/ORL"
subjects = 40
images_per_subject = 7  # Changed for new split (7 for training, 3 for testing)
test_size = 0.3  # New test size for train_test_split

def load_data(data_path, num_subjects, images_per_subject, image_size):
    D = []
    labels = []
    for subject_id in range(1, num_subjects + 1):
        subject_folder = os.path.join(data_path, f"s{subject_id}")
        for img_idx in range(1, images_per_subject + 1):
            img_path = os.path.join(subject_folder, f"{img_idx}.pgm")
            img = io.imread(img_path)
            img_data = img.flatten()  # Flatten image
            D.append(img_data)
            labels.append(subject_id)
    return np.array(D), np.array(labels)

# Load data
D, labels = load_data(dataset_path, subjects, images_per_subject, image_size=(92, 112))

# Split data using train_test_split (alternative to odd/even split)
X_train, X_test, y_train, y_test = train_test_split(D, labels, test_size=test_size, random_state=42)


y_train = y_train.astype('int')
y_test = y_test.astype('int')


# LDA Algorithm

# Calculate overall sample mean
mu = np.mean(X_train, axis=0)


# Calculate mean for each class and store them in a vector6
mean_vector = np.array([np.mean(X_train[y_train == i], axis=0) for i in range(1, 41)])

# Calculate within-class scatter matrix  (Sw)
Sw = np.zeros((X_train.shape[1], X_train.shape[1]))
for subject_id in range(1, subjects + 1):
    class_scatter = np.cov(D[labels == subject_id].T)
    Sw += class_scatter * (images_per_subject -1)



# Calculate between-class scatter matrix  (Sb)
Sb = np.zeros((X_train.shape[1],X_train.shape[1]))
for mv in mean_vector:
    diff = mv - mu
    Sb += images_per_subject * np.outer(diff, diff)


# find eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))  #sw(inverse) sb * v = lambda * v


sorted_indices = np.argsort(eigenvalues)[::-1] # Sort eigenvectors by eigenvalues in from largest lambda to smallest lambda
U = eigenvectors[:, sorted_indices[:39]] 
U = U.real # make sure all values in eigen vector are real numbers


# dimension reduction on test data and training data by multiplying with U (the result eigen vector)
X_train_lda = X_train.dot(U)
X_test_lda = X_test.dot(U)


# Train a KNN Model 
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_lda, y_train)
y_pred = knn_classifier.predict(X_test_lda)


# calc. accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Multiclass LDA: {accuracy:.2f}")

