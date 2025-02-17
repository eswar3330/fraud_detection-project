import pandas as pd
import numpy as np
import logging
import time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load dataset
logging.info("Loading dataset...")
df = pd.read_csv("creditcard.csv")

# Drop 'Time' column
df.drop(columns=["Time"], inplace=True)

# Separate features and target
X = df.drop(columns=["Class"])
y = df["Class"]

# Normalize numerical features
logging.info("Normalizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

# Apply SMOTE for class balance
logging.info("Applying SMOTE for class balance...")
smote = SMOTE(sampling_strategy=0.7, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train Isolation Forest
logging.info("Training Isolation Forest...")
iso_forest = IsolationForest(n_estimators=300, contamination=0.01, random_state=42, max_features=0.9)
iso_forest.fit(X_train)
joblib.dump(iso_forest, "isolation_forest.pkl")

# Train Autoencoder
logging.info("Training Autoencoder...")
input_dim = X_train.shape[1]
autoencoder = Sequential([
    Dense(64, activation="relu", input_shape=(input_dim,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(32, activation="relu"),
    BatchNormalization(),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(input_dim, activation="sigmoid")
])
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(X_train[y_train == 0], X_train[y_train == 0], epochs=30, batch_size=64, validation_split=0.1)
autoencoder.save("autoencoder_model.h5")

# Evaluate models
def evaluate_model(model, X_test, y_test, is_autoencoder=False):
    try:
        if is_autoencoder:
            X_pred = model.predict(X_test)
            reconstruction_error = np.mean(np.abs(X_test - X_pred), axis=1)
            y_pred = np.where(reconstruction_error > 0.005, 1, 0)
        else:
            y_pred = model.predict(X_test)
            y_pred = np.where(y_pred == -1, 1, 0)

        logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        logging.info(f"ROC AUC Score: {roc_auc_score(y_test, y_pred)}")

    except Exception as e:
        logging.error(f"⚠️ Error evaluating model: {e}")

# Evaluate Isolation Forest
evaluate_model(iso_forest, X_test, y_test)

# Evaluate Autoencoder
evaluate_model(autoencoder, X_test, y_test, is_autoencoder=True)

# Batch Processing Fraud Detection
def process_batch_data(batch_size=100):
    try:
        batch_start = 0
        while batch_start < X_test.shape[0]:
            batch_end = min(batch_start + batch_size, X_test.shape[0])
            X_batch = X_test[batch_start:batch_end]
            y_batch = y_test[batch_start:batch_end]

            # Process Isolation Forest for the batch
            iso_pred = iso_forest.predict(X_batch)
            iso_pred = np.where(iso_pred == -1, 1, 0)

            # Process Autoencoder for the batch
            auto_pred = autoencoder.predict(X_batch)
            reconstruction_error = np.mean(np.abs(X_batch - auto_pred), axis=1)
            auto_pred = np.where(reconstruction_error > 0.005, 1, 0)

            # Log results for batch
            logging.info(f"Batch {batch_start // batch_size + 1} results:")
            logging.info(f"Isolation Forest Prediction: {iso_pred}")
            logging.info(f"Autoencoder Prediction: {auto_pred}")

            # Optionally, you could log detailed classification metrics per batch
            logging.info(f"Confusion Matrix for Batch {batch_start // batch_size + 1} (Isolation Forest):\n{confusion_matrix(y_batch, iso_pred)}")
            logging.info(f"Classification Report for Batch {batch_start // batch_size + 1} (Isolation Forest):\n{classification_report(y_batch, iso_pred)}")
            logging.info(f"Confusion Matrix for Batch {batch_start // batch_size + 1} (Autoencoder):\n{confusion_matrix(y_batch, auto_pred)}")
            logging.info(f"Classification Report for Batch {batch_start // batch_size + 1} (Autoencoder):\n{classification_report(y_batch, auto_pred)}")

            # Move to the next batch
            batch_start += batch_size
            time.sleep(1)  # Sleep for 1 second before processing the next batch

    except Exception as e:
        logging.error(f"⚠️ Error in batch processing: {e}")

# Start batch processing fraud detection
try:
    logging.info("Starting batch processing fraud detection...")
    process_batch_data(batch_size=1000)  # Process in batches of 100
except KeyboardInterrupt:
    logging.info("Batch processing stopped.")

logging.info("✅ Batch fraud detection completed!")
