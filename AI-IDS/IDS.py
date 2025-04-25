import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import randint

# Data Loading and Preprocessing
columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", 
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", 
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", 
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", 
    "dst_host_serror_rate", "dst_host_srv_serror_rate", 
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

train_data = pd.read_csv("KDDTrain+.txt", header=None, names=columns)
test_data = pd.read_csv("KDDTest+.txt", header=None, names=columns)

# Drop difficulty column as it's not needed
train_data.drop("difficulty", axis=1, inplace=True)
test_data.drop("difficulty", axis=1, inplace=True)

# One-hot encode categorical columns
categorical_columns = ["protocol_type", "service", "flag"]
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_train = pd.DataFrame(encoder.fit_transform(train_data[categorical_columns]))
encoded_test = pd.DataFrame(encoder.transform(test_data[categorical_columns]))

# Drop original categorical columns and replace them with the encoded ones
train_data.drop(columns=categorical_columns, inplace=True)
test_data.drop(columns=categorical_columns, inplace=True)

# Normalization (Scaling)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train_data.drop("label", axis=1))
X_test = scaler.transform(test_data.drop("label", axis=1))

# Binary classification (0 for normal, 1 for attacks)
y_train = train_data["label"].apply(lambda x: 0 if x == "normal" else 1).astype(int)
y_test = test_data["label"].apply(lambda x: 0 if x == "normal" else 1).astype(int)

# ---- Random Forest Hyperparameter Tuning with RandomizedSearchCV ----

# Define parameter grid for Random Forest
rf_param_dist = {
    'n_estimators': randint(200, 500),  # Increase the number of trees
    'max_depth': [10, 20, 30, 50, None],  # Additional depth options
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Randomized Search for Random Forest with more iterations and cross-validation
rf_random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), rf_param_dist, n_iter=20, cv=3, n_jobs=-1, scoring='accuracy', random_state=42)
rf_random_search.fit(X_train, y_train)

# Best Random Forest model
best_rf_model = rf_random_search.best_estimator_

y_pred_rf = best_rf_model.predict(X_test)
print("Optimized Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# ---- Neural Network Hyperparameter Tuning ----

# Define improved Neural Network model with more layers and epochs
nn_model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid') 
])

# Compile the model
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Add Early Stopping to monitor validation accuracy and stop if it stops improving
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with more epochs
nn_model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=2, callbacks=[early_stopping])

# Evaluate the neural network model
loss, accuracy = nn_model.evaluate(X_test, y_test)
print("Neural Network Test Accuracy:", accuracy)


# ---- Adjustments for Desired Accuracy ----

# If accuracy exceeds 90%, consider reducing epochs, or increasing dropout in the neural network to avoid overfitting.
# Similarly, fine-tune Random Forest further or introduce additional preprocessing techniques if needed.
