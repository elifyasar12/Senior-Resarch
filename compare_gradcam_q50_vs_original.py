import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2
import os

# === CONFIG ===
img_paths = {
    "Original": "dataset/original/valid/fracture/fracture1.jpg",
    "JPEG 50": "dataset/q50/valid/fracture/fracture1.jpg"
}
model_paths = {
    "Original": "resnet50_original.h5",
    "JPEG 50": "resnet50_q50.h5"
}
img_size = (224, 224)

# === Grad-CAM Function ===
def generate_gradcam(model_path, img_path):
    model = tf.keras.models.load_model(model_path)
    last_conv_layer_name = "conv5_block3_out"  # Last conv layer in ResNet50

    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    img_cv = cv2.imread(img_path)
    img_cv = cv2.resize(img_cv, img_size)
    heatmap_resized = cv2.resize(heatmap, img_size)

    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)

    return cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

# === Generate Heatmaps ===
outputs = []
titles = []

for label in img_paths:
    gradcam_image = generate_gradcam(model_paths[label], img_paths[label])
    outputs.append(gradcam_image)
    titles.append(label)

# === Plot Side-by-Side ===
fig, axes = plt.subplots(1, len(outputs), figsize=(10, 4))
for i, ax in enumerate(axes):
    ax.imshow(outputs[i])
    ax.set_title(titles[i])
    ax.axis('off')

plt.tight_layout()
plt.savefig("gradcam_comparison_q50_vs_original.png")
plt.show()
