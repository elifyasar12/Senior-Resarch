from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

model_path = "resnet50_q90.h5"  # change to q70, q50, etc. if needed
model = load_model(model_path)

test_dir = "resnet_test_data"

for label in ["fracture", "no_fracture"]:
    folder = os.path.join(test_dir, label)
    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder, file)
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_array)[0][0]
            prediction = "Fracture" if pred > 0.5 else "No Fracture"
            print(f"{file} → Predicted: {prediction} ({pred:.2f})")
