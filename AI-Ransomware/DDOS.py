import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

# Load the dataset
df = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

# Clean up the column names (strip any leading/trailing spaces)
df.columns = df.columns.str.strip()

# Check the column names to ensure 'Label' exists
print("Columns in the dataset:", df.columns)

# Ensure the target column is 'Label'
target_column = 'Label'  # Update this if needed based on column inspection

# Encode the target labels
if target_column in df.columns:
    label_encoder = LabelEncoder()
    df[target_column] = label_encoder.fit_transform(df[target_column])
else:
    raise ValueError(f"Target column '{target_column}' not found in the dataset. Please check the column names.")

# Keep the target column and select only the numeric columns
df_numeric = df.select_dtypes(include=['float64', 'int64'])

# Make sure the target column is present in the numeric DataFrame
if target_column not in df_numeric.columns:
    df_numeric[target_column] = df[target_column]  # Add the target column back if missing

# Split the data into features and labels
X = df_numeric.drop(columns=[target_column])  # Features
y = df_numeric[target_column]  # Target

# Handle NaN and infinite values
# Replace NaN values with the column mean
X = X.replace([np.inf, -np.inf], np.nan)  # Replace infinity values with NaN
X = X.fillna(X.mean())  # Replace NaN with the mean of each column

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Data Balancing with SMOTE (using a more conservative resampling ratio)
smote = SMOTE(sampling_strategy=0.4, random_state=42)  # Less aggressive resampling
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split into training and testing sets with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# 1. Deep Learning Model with Regularization (Dropout)
dl_model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.5),  # Increased Dropout rate for stronger regularization
    Dense(32, activation='relu'),
    Dropout(0.5),  # Increased Dropout rate for stronger regularization
    Dense(16, activation='relu'),
    Dropout(0.5),  # Increased Dropout rate for stronger regularization
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model with a lower learning rate
dl_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the deep learning model with fewer epochs
dl_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate the deep learning model
dl_loss, dl_accuracy = dl_model.evaluate(X_test, y_test)
print(f"Deep Learning Model Accuracy: {dl_accuracy * 100:.2f}%")

# 2. Logistic Regression with Cross-Validation and Hyperparameter Tuning
lr_model = LogisticRegression(max_iter=1000, random_state=42)

# Set up GridSearch for Logistic Regression hyperparameters
param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['liblinear', 'saga']
}

grid_search = GridSearchCV(lr_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters for Logistic Regression
print(f"Best Parameters for Logistic Regression: {grid_search.best_params_}")

# Evaluate the Logistic Regression model with the best parameters
best_lr_model = grid_search.best_estimator_
y_pred_lr = best_lr_model.predict(X_test)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
class_report_lr = classification_report(y_test, y_pred_lr)

# Print evaluation metrics for Logistic Regression
print(f"\nLogistic Regression Accuracy: {accuracy_lr * 100:.2f}%")
print("\nConfusion Matrix (Logistic Regression):\n", conf_matrix_lr)
print("\nClassification Report (Logistic Regression):\n", class_report_lr)

# 3. Random Forest Classifier with Cross-Validation
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Cross-validation for Random Forest
rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"\nRandom Forest Classifier Cross-Validation Accuracy: {rf_cv_scores.mean() * 100:.2f}%")

# Train the Random Forest model and evaluate
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
class_report_rf = classification_report(y_test, y_pred_rf)

# Print evaluation metrics for Random Forest
print(f"\nRandom Forest Classifier Accuracy: {accuracy_rf * 100:.2f}%")
print("\nConfusion Matrix (Random Forest):\n", conf_matrix_rf)
print("\nClassification Report (Random Forest):\n", class_report_rf)




