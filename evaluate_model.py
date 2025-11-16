"""
import kagglehub

# Download latest version
path = kagglehub.dataset_download("davilsena/ckdataset")

print("Path to dataset files:", path)
"""
# Path to dataset files: /home/polperez/.cache/kagglehub/datasets/davilsena/ckdataset/versions/2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import load_model
import keras
import pickle
import time

total_time = time.time()
# -----------------------------
# 1. Load CK+ dataset CSV
# -----------------------------
csv_path = '/home/polperez/.cache/kagglehub/datasets/davilsena/ckdataset/versions/2/ckextended.csv'
ck_df = pd.read_csv(csv_path)

# Select only test data
test_df = ck_df[ck_df['Usage'].isin(['PublicTest', 'PrivateTest'])]

# Extract labels and pixels
y_test = test_df['emotion'].values
X_test = np.array([np.fromstring(pix, sep=' ') for pix in test_df['pixels']], dtype='float32')

# Normalize and reshape
X_test = X_test / 255.0
X_test = X_test.reshape(-1, 48, 48, 1)

# -----------------------------
# 2. Filter test data to match trained emotions (0-6)
# -----------------------------
mask = y_test <= 6
X_test = X_test[mask]
y_test = y_test[mask]

print(f"Filtered test images: {X_test.shape}, Filtered test labels: {y_test.shape}")

# -----------------------------
# 3. Load trained CNN model
# -----------------------------
model = keras.models.load_model("/home/polperez/Desktop/final_emotion_recognition_cnn.keras")

# -----------------------------
# 4. Create dummy second input
# -----------------------------
X_dummy = np.zeros((X_test.shape[0], 136), dtype=np.float32)  # match second input size



# -----------------------------
# 6. Predictions
# -----------------------------
y_pred_prob = model.predict([X_test, X_dummy])
y_pred = np.argmax(y_pred_prob, axis=1)
"""
# -----------------------------
# 7. Dataset distribution
# -----------------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
pd.Series(y_test).value_counts().sort_index().plot(kind='bar')
plt.title("True Samples per Emotion")
plt.xlabel("Emotion Label")
plt.ylabel("Count")

plt.subplot(1,2,2)
pd.Series(y_pred).value_counts().sort_index().plot(kind='bar')
plt.title("Predicted Samples per Emotion")
plt.xlabel("Emotion Label")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# -----------------------------
# 8. Confusion matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# -----------------------------
# 9. Classification report
# -----------------------------
report = classification_report(y_test, y_pred, digits=4)
print("Classification Report:\n", report)

# -----------------------------
# 10. ROC curves & AUC
# -----------------------------
n_classes = len(np.unique(y_test))
y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))

plt.figure(figsize=(8,6))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC={roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # diagonal
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves per Class')
plt.legend()
plt.show()

end_total_time = time.time()
time = end_total_time - total_time
print(f"\nTotal time: {time:.3f} seconds")


# -----------------------------
# 11. Training accuracy & loss over epochs (if history saved)
# -----------------------------
history_path = '/home/polperez/Desktop/final_model_history.pkl'  # adjust path
try:
    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    # Accuracy
    plt.figure(figsize=(8,4))
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Loss
    plt.figure(figsize=(8,4))
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

except FileNotFoundError:
    print("Training history file not found. Skipping accuracy/loss plots.")

from data_preparing_for_CNN import history  # make sure training_script.py is in the same folder


# Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

import pickle
import matplotlib.pyplot as plt

# --- Load the training history ---
with open("/home/polperez/Desktop/final_model_history.pkl", "rb") as f:
    history = pickle.load(f)

# --- Print available metrics ---
print("Available keys in history:", history.keys())

# --- Plot Training vs Validation Accuracy ---
plt.figure(figsize=(10,5))
plt.plot(history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title("Model Accuracy over Epochs", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.legend()
plt.grid(True)
plt.show(block=True)   # keeps figure open until you close it manually

# --- Plot Training vs Validation Loss ---
plt.figure(figsize=(10,5))
plt.plot(history['loss'], label='Training Loss', linewidth=2)
plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
plt.title("Model Loss over Epochs", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend()
plt.grid(True)
plt.show(block=True)   # same behavior: stays open until you close it

# -----------------------------
# 5. Evaluate loss & accuracy
# -----------------------------
import numpy as np
import matplotlib.pyplot as plt

batch_size = 32
num_batches = int(np.ceil(len(X_test) / batch_size))

test_losses = []
test_accuracies = []

for i in range(num_batches):
    X_batch = X_test[i*batch_size:(i+1)*batch_size]
    y_batch = y_test[i*batch_size:(i+1)*batch_size]
    loss, acc = model.evaluate([X_batch, X_dummy[i*batch_size:(i+1)*batch_size]], y_batch, verbose=0)
    test_losses.append(loss)
    test_accuracies.append(acc)

# Plot testing performance
plt.figure(figsize=(8,5))
plt.plot(test_losses, label='Test Loss', color='red')
plt.plot(test_accuracies, label='Test Accuracy', color='blue')
plt.xlabel('Batch number')
plt.ylabel('Value')
plt.title('Test Loss and Accuracy per Batch')
plt.legend()
plt.grid(True)
plt.show(block=True)

"""