import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE  # For balancing classes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load and preprocess data
df = pd.read_csv('SQLiV3.csv')

# Fill missing values
df['query'] = df['query'].fillna('')  # Fill NaN in 'query' column with empty strings
df['Label'] = df['Label'].fillna(0)   # Fill NaN in 'Label' column with 0

# Map string labels to integers if necessary
df['Label'] = df['Label'].map({'benign': 0, 'sql_injection': 1})

# Feature and target columns
X = df['query']
y = df['Label']

# Encode target labels (if needed)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Check the original class distribution
print(f"Class distribution in original data: {np.bincount(y)}")

# Split into train and test sets with stratified sampling to ensure both classes are represented
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert text data to numerical vectors using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Convert sparse matrices to dense
X_train_tfidf = X_train_tfidf.toarray()
X_test_tfidf = X_test_tfidf.toarray()

# Check class distribution in training set
print(f"Class distribution in y_train: {np.bincount(y_train)}")
print(f"Class distribution in y_test: {np.bincount(y_test)}")

# Handle class imbalance with SMOTE only if both classes are present
if len(np.unique(y_train)) > 1:  # Ensure that there are both classes in the target
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_tfidf, y_train)
    print(f"Class distribution after SMOTE in y_train_res: {np.bincount(y_train_res)}")
else:
    print("Warning: Only one class in training data. SMOTE is not applied.")
    X_train_res, y_train_res = X_train_tfidf, y_train  # Use the original data if one class is present

# Logistic Regression Model with class_weight to handle class imbalance
lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_model.fit(X_train_res, y_train_res)

# Predict and evaluate Logistic Regression
y_pred_lr = lr_model.predict(X_test_tfidf)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

print(f"Logistic Regression Accuracy: {accuracy_lr * 100:.2f}%")
print(f"Precision: {precision_lr * 100:.2f}%")
print(f"Recall: {recall_lr * 100:.2f}%")
print(f"F1-score: {f1_lr * 100:.2f}%")

# Cross-validation for more reliable results
cross_val_scores = cross_val_score(lr_model, X_train_tfidf, y_train_res, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy (Logistic Regression): {cross_val_scores.mean() * 100:.2f}%")

# Confusion Matrix for Logistic Regression
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
print("Logistic Regression Confusion Matrix:\n", conf_matrix_lr)
