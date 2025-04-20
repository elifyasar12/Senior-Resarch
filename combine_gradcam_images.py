from PIL import Image
import matplotlib.pyplot as plt

# Load your Grad-CAMs
fracture_img = Image.open("Figure_2.png")       # Grad-CAM for fracture
no_fracture_img = Image.open("Figure_3.png")    # Grad-CAM for no fracture

# Resize (if needed) to match
fracture_img = fracture_img.resize((300, 300))
no_fracture_img = no_fracture_img.resize((300, 300))

# Combine side-by-side
combined_width = fracture_img.width + no_fracture_img.width
combined = Image.new("RGB", (combined_width, fracture_img.height))
combined.paste(fracture_img, (0, 0))
combined.paste(no_fracture_img, (fracture_img.width, 0))

# Save
output_path = "combined_gradcam.png"
combined.save(output_path)
print(f"✅ Combined Grad-CAM saved as: {output_path}")

# Display
plt.imshow(combined)
plt.axis('off')
plt.title("Grad-CAM Comparison: Fracture vs No Fracture")
plt.show()
