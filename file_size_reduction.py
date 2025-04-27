import os
import matplotlib.pyplot as plt

# Paths to your images
original_image = "dataset/original/train/fracture/fracture1.jpg"
jpeg90_image = "dataset/q90/train/fracture/fracture1.jpg"
jpeg70_image = "dataset/q70/train/fracture/fracture1.jpg"
jpeg50_image = "dataset/q50/train/fracture/fracture1.jpg"

# Function to get file size in KB
def get_file_size(path):
    return os.path.getsize(path) / 1024  # Convert bytes to KB

# Get file sizes
original_size = get_file_size(original_image)
sizes = {
    "JPEG 100 (Original)": original_size,
    "JPEG 90": get_file_size(jpeg90_image),
    "JPEG 70": get_file_size(jpeg70_image),
    "JPEG 50": get_file_size(jpeg50_image)
}

# 📦 Print file sizes and reductions
print("\n📦 File Size Reduction Report:")
for quality, size in sizes.items():
    if quality != "JPEG 100 (Original)":
        reduction = (1 - size / original_size) * 100
        print(f"{quality}: {size:.2f} KB  |  Reduction: {reduction:.1f}%")
    else:
        print(f"{quality}: {size:.2f} KB (baseline)")

# 📈 Plotting
qualities = list(sizes.keys())
values = list(sizes.values())

plt.figure(figsize=(8,5))
bars = plt.bar(qualities, values, color=["blue", "green", "orange", "red"])
plt.ylabel("File Size (KB)")
plt.title("File Size Reduction with JPEG Compression")

# Annotate bars with exact size
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f"{yval:.1f} KB", ha='center', va='bottom')

plt.tight_layout()
plt.savefig("file_size_reduction.png")
plt.show()

print("\n✅ Bar chart saved as file_size_reduction.png!")
