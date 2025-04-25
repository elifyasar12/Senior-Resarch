from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc, f1_score
import numpy as np
import os

# === CONFIG ===
model_path = "resnet50_q90.h5"
dataset_path = "dataset/q90/valid"
image_size = (224, 224)
batch_size = 8

# === Load Model ===
model = load_model(model_path)

# === Data Generator ===
datagen = ImageDataGenerator(rescale=1./255)
val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# === Get Predictions ===
pred_probs = model.predict(val_gen)
preds = (pred_probs > 0.5).astype(int)
true_labels = val_gen.classes

# === Metrics ===
f1 = f1_score(true_labels, preds)
auc_roc = roc_auc_score(true_labels, pred_probs)
precision, recall, _ = precision_recall_curve(true_labels, pred_probs)
auc_pr = auc(recall, precision)

print("\n📊 Performance Report:")
print(f"F1 Score: {f1:.2f}")
print(f"AUROC: {auc_roc:.2f}")
print(f"AUPRC: {auc_pr:.2f}")

print("\nClassification Report:")
print(classification_report(true_labels, preds, target_names=['no_fracture', 'fracture']))

print("Confusion Matrix:")
print(confusion_matrix(true_labels, preds))
