from skimage import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


dataset = "C:/Users/ahmed/OneDrive/Desktop/FaceRegocnition/ORL" 
subjects = 40           
images_per_subject = 10   
image_size = (92, 112) 


# Initialize empty arrays for data matrix and label vector
D = np.zeros((subjects * images_per_subject , image_size[0] * image_size[1] ))   # array D (rows, columns) as (400 rows, 10304 columns)
y = np.zeros(subjects * images_per_subject, dtype=int)                         # array y to label rows 


for subject_id in range(1, subjects + 1):                                  # Read subjects from ORL folder from s1 to s40
    subject_folder = os.path.join(dataset, f"s{subject_id}")               # make a path for each folder
    for img_idx in range(1, images_per_subject + 1):                       # read each folder's 10 images
        img_path = os.path.join(subject_folder, f"{img_idx}.pgm")          #make a path for each image in (.pgm format) 
        img = io.imread(img_path)                                          #reads image path
        
        D[(subject_id - 1) * images_per_subject + img_idx - 1, :] = img.flatten() # 1D vector of each image is added to D array as a column
        y[(subject_id - 1) * images_per_subject + img_idx - 1] = subject_id       # adds the label to subject which is an interger from 1:40


X_train = D[0::2]  
y_train = y[0::2] 
X_test = D[1::2]  
y_test = y[1::2]  


print("Training Data Matrix shape:", X_train.shape)
print("Training Label Vector shape:", y_train.shape)
print("Test Data Matrix shape:", X_test.shape)
print("Test Label Vector shape:", y_test.shape)
print(y_train)


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
    class_scatter = np.cov(D[y == subject_id].T)
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





