# train_resnet50_by_quality.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import os
import matplotlib.pyplot as plt

base_dir = "dataset"
folders = ["original", "q90", "q70", "q50"]
qualities = [100, 90, 70, 50]
image_size = (224, 224)
batch_size = 8
epochs = 5
results = {}

def create_model():
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base.input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

for folder, q in zip(folders, qualities):
    print(f"\n🔍 Training on: {folder} (JPEG {q})")
    path = os.path.join(base_dir, folder)

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        os.path.join(path, 'train'),
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )
    val_gen = datagen.flow_from_directory(
        os.path.join(path, 'valid'),
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    model = create_model()
    model.fit(train_gen, validation_data=val_gen, epochs=epochs, verbose=1)

    loss, acc = model.evaluate(val_gen)
    results[q] = acc * 100
    model.save(f"resnet50_q{q}.h5")
    print(f"✅ JPEG {q} Accuracy: {acc*100:.2f}%")

plt.figure(figsize=(8,5))
plt.plot(sorted(results.keys(), reverse=True), [results[k] for k in sorted(results.keys(), reverse=True)], marker='o')
plt.title("ResNet50 Accuracy vs JPEG Quality")
plt.xlabel("JPEG Quality")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig("accuracy_vs_jpeg_graph.png")
plt.show()
