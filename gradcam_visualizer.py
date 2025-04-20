import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2
import os

# === CONFIG ===
model_path = "resnet50_q100.h5"
img_path = "dataset/resnet_test_data/fracture/Broken Bone 1.jpeg"  # Change this to test other images
img_size = (224, 224)

# === Load model and image ===
model = tf.keras.models.load_model(model_path)
img = image.load_img(img_path, target_size=img_size)
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# === Get model prediction ===
pred = model.predict(img_array)[0][0]
pred_class = "Fracture" if pred > 0.5 else "No Fracture"
print(f"Prediction: {pred_class} ({pred:.2f})")

# === Get last convolutional layer ===
last_conv_layer_name = "conv5_block3_out"  # For ResNet50

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

# === Convert to color map and overlay ===
img_cv = cv2.imread(img_path)
img_cv = cv2.resize(img_cv, img_size)
heatmap_resized = cv2.resize(heatmap, img_size)
heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)

# === Save and display ===
output_path = "gradcam_output.png"
cv2.imwrite(output_path, superimposed_img)
print(f"Grad-CAM saved to: {output_path}")

# Show in notebook/preview
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.title(f"{pred_class} ({pred:.2f})")
plt.axis('off')
plt.show()
