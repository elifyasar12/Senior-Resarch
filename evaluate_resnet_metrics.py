from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc, f1_score, accuracy_score
import numpy as np
import os

# Define compression quality tags (dataset folders and model suffixes)
qualities = ['original', 'q90', 'q70', 'q50']
image_size = (224, 224)
batch_size = 8

print("\n🔁 Evaluating ResNet50 models on all JPEG qualities:\n")
print(f"{'JPEG':<9} | {'F1':<6} | {'AUROC':<6} | {'AUPRC':<6} | {'Acc':<6}")
print("-" * 45)

for q in qualities:
    model_path = f"resnet50_{q}.h5"
    dataset_path = f"dataset/{q}/valid"

    model = load_model(model_path)

    datagen = ImageDataGenerator(rescale=1./255)
    val_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    pred_probs = model.predict(val_gen)
    preds = (pred_probs > 0.5).astype(int)
    true_labels = val_gen.classes

    f1 = f1_score(true_labels, preds)
    auc_roc = roc_auc_score(true_labels, pred_probs)
    precision, recall, _ = precision_recall_curve(true_labels, pred_probs)
    auc_pr = auc(recall, precision)
    acc = accuracy_score(true_labels, preds)

    print(f"{q:<9} | {f1:.2f}  | {auc_roc:.2f}  | {auc_pr:.2f}  | {acc:.2f}")

print("\n✅ Evaluation complete.")
