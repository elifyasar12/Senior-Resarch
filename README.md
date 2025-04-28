Bone Fracture Detection with JPEG Compression

Overview

This project focuses on bone fracture detection in X-ray images using deep learning models, specifically exploring the impact of JPEG compression on image quality and model performance. The goal is to understand the trade-off between compression (for storage efficiency) and image quality (for accurate fracture detection).

Project Structure

Copy
├── dataset/ # Folder containing the dataset of X-ray images
│ ├── original/ # High-quality images (JPEG quality 100)
│ ├── q50/ # JPEG quality 50 images
│ ├── q70/ # JPEG quality 70 images
│ ├── q90/ # JPEG quality 90 images
├── scripts/ # Python scripts for data processing and analysis
│ ├── accuracy_vs_jpeg.py # Script for analyzing accuracy vs JPEG quality
│ ├── file_size_reduction.py # Script for analyzing file size reductions based on compression levels
│ ├── evaluate_psnr_ssim.py # Script for evaluating PSNR and SSIM for image quality
│ ├── train_resnet50_by_quality.py # Script for training the fracture detection model with JPEG quality levels
│ ├── gradcam_comparison.py # Script for comparing Grad-CAM results at different JPEG qualities
│ ├── gradcam_visualize.py # Script for visualizing Grad-CAM heatmaps
├── requirements.txt # List of required Python dependencies
├── README.md # This README file
└── results/ # Folder for storing analysis and model output

Installation

- Dependencies
  This project uses Python 3.x and virtual environments to isolate dependencies. Here’s how to set up your environment:

- Create a virtual environment: In your project directory, run the following command:

python3 -m venv venv310

- On macOS/Linux:
  source venv310/bin/activate

- n Windows:
  venv310\Scripts\activate

- requirements.txt
  The requirements.txt file should contain the following libraries:

matplotlib: for generating graphs and visualizations

numpy: for numerical operations

tensorflow or keras: for deep learning model training

os: for directory management

sys: for system-level operations

- Directory Structure
  Ensure your dataset is structured correctly inside the dataset/ folder, with subfolders for different JPEG quality levels:

original: High-quality images (JPEG quality 100)

q50: JPEG quality 50 images

q70: JPEG quality 70 images

q90: JPEG quality 90 images

Usage

- File Size Analysis
  To analyze the impact of JPEG compression on file size, run the file_size_reduction.py script. This will generate a bar chart comparing the average file sizes at different compression levels (100, 90, 70, 50).

python scripts/file_size_reduction.py

- Model Training
  To train the bone fracture detection model using ResNet50 on images at different JPEG compression levels, use the train_resnet50_by_quality.py script. This will preprocess the dataset, train the model, and output performance metrics.

python scripts/train_resnet50_by_quality.py

- Accuracy vs Compression
  To visualize how JPEG compression affects model accuracy, use the accuracy_vs_jpeg.py script, which compares the accuracy at various quality levels.

python scripts/accuracy_vs_jpeg.py

- Grad-CAM Visualization
  To visualize the Grad-CAM results for the model’s decision-making, use the gradcam_comparison.py and gradcam_visualize.py scripts. These scripts compare and visualize heatmaps for different compression levels.

python scripts/gradcam_comparison.py
python scripts/gradcam_visualize.py

-PSNR and SSIM Evaluation
To evaluate the image quality using PSNR and SSIM after applying JPEG compression, run the evaluate_psnr_ssim.py script.
python scripts/evaluate_psnr_ssim.py

Key Additions:
Virtual Environment: Instructions on creating and activating a virtual environment.

Dependencies: Details about installing the necessary Python packages using pip.

Script Descriptions: Explanation of what each script does and how to run them.
