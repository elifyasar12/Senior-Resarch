#!/usr/bin/env python3
import os
import sys
import matplotlib.pyplot as plt

# 1) Find script directory and dataset folder
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(script_dir, "dataset")
if not os.path.isdir(dataset_dir):
    print(f"ERROR: dataset/ folder not found at {dataset_dir}", file=sys.stderr)
    sys.exit(1)

# 2) Define the subfolders for each JPEG quality
qualities = {
    "100 (orig)": "original",
    "90":         "q90",
    "70":         "q70",
    "50":         "q50",
}

avg_sizes = []
labels    = []

# 3) Walk each folder, gather .jpg/.jpeg sizes, compute average
for label, sub in qualities.items():
    folder = os.path.join(dataset_dir, sub)
    if not os.path.isdir(folder):
        print(f"Warning: {folder} does not exist, skipping", file=sys.stderr)
        continue

    sizes = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg")):
                full = os.path.join(root, f)
                sizes.append(os.path.getsize(full))

    if not sizes:
        print(f"Warning: no JPEG files found under {folder}", file=sys.stderr)
        continue

    avg = sum(sizes) / len(sizes)
    avg_sizes.append(avg / 1024)  # convert to KB
    labels.append(label)

# 4) Plot
plt.figure(figsize=(6,4))
bars = plt.bar(labels, avg_sizes)
plt.ylabel("Average File Size (KB)")
plt.xlabel("JPEG Quality")
plt.title("Average X-ray File Size by JPEG Compression Level")

# annotate bars
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f} KB",
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()
