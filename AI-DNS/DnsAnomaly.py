from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def load_dns_logs(csv_file):
    data = pd.read_csv(csv_file)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    return data

def preprocess_features(data):
    data["hour"] = data["timestamp"].dt.hour
    data["transaction_id_str"] = data["transaction_id"].astype(str)
    
    entropy_approx = (
        data.groupby("hour")["transaction_id_str"]
        .apply(lambda x: len(set(x)) / len(x))
    )
    data["transaction_id_entropy"] = data["hour"].map(entropy_approx)
    
    features = data[["transaction_id", "ttl"]].copy()
    features["transaction_id_entropy"] = data["transaction_id_entropy"]
    features.fillna(0, inplace=True)
    return features



def train_models(features, labels=None):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    features["iso_anomaly"] = iso_forest.fit_predict(scaled_features)
    features["iso_anomaly"] = features["iso_anomaly"].apply(lambda x: 1 if x == -1 else 0)
    
    # One-Class SVM
    one_class_svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='auto')
    features["svm_anomaly"] = one_class_svm.fit_predict(scaled_features)
    features["svm_anomaly"] = features["svm_anomaly"].apply(lambda x: 1 if x == -1 else 0)


    if labels is not None:
        print("Isolation Forest Results")
        print_metrics(labels, features["iso_anomaly"])
        print("\nOne-Class SVM Results")
        print_metrics(labels, features["svm_anomaly"])
    
    return iso_forest, one_class_svm, features



def print_metrics(labels, predictions):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")



csv_file = "synthetic_dns_logs.csv"
dns_logs = load_dns_logs(csv_file)
features = preprocess_features(dns_logs)

labels = np.random.choice([0, 1], size=len(features), p=[0.9, 0.1])

# Train both models and evaluate accuracy
iso_model, svm_model, features_with_anomalies = train_models(features, labels)

# Output the anomaly records detected by both models
iso_anomalies = dns_logs[features_with_anomalies["iso_anomaly"] == 1]
svm_anomalies = dns_logs[features_with_anomalies["svm_anomaly"] == 1]

print(f"\nIsolation Forest detected {len(iso_anomalies)} anomalies.")
print(f"One-Class SVM detected {len(svm_anomalies)} anomalies.")
