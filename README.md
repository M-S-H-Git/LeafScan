# LeafScan: Plant Disease Detection System

> An intelligent, machine-learning-based application for automated plant health analysis and disease diagnosis.

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-orange)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-OpenCV-blue)
![Accuracy](https://img.shields.io/badge/Validation%20Accuracy-94.5%25-brightgreen)

## Overview
Plant diseases are a major cause of reduced agricultural productivity worldwide. Traditional identification relies heavily on manual inspection, which is time-consuming, costly, and often inaccessible to farmers. 

LeafScan addresses this problem by utilizing Artificial Intelligence and Machine Learning to automatically detect plant diseases from leaf images. By extracting color, texture, and shape characteristics, the system provides fast, accurate, and cost-effective disease diagnosis, helping farmers take timely preventive measures.

[📄 **Read the Full Technical Report (PDF)**](./LeafScan_Report.pdf)

## Key Features
* **Automated Disease Classification:** Accurately identifies multiple conditions including Early Blight, Late Blight, Leaf Mold, Bacterial Spot, Septoria Leaf Spot, and Healthy leaves.
* **Lesion Annotation:** Visually highlights infected areas and lesions on the uploaded leaf image.
* **Actionable Insights:** Provides confidence scores, severity ratings, recommended treatments, and preventative measures for detected diseases.
* **High Accuracy:** Achieves **94.56% overall validation accuracy** utilizing an optimized Ensemble Random Forest Classifier.

## System Architecture & Pipeline

The system is built on a multi-tier architecture to process images efficiently from the client to the machine learning model:

1. **Client Tier (Frontend UX):** JavaScript-based interface allowing users to upload leaf images via Drag & Drop, sending a `FormData` payload via HTTP POST.
2. **Processing Tier (Computer Vision):** 
   * **OpenCV Matrix Transmutation:** Resizes images (256x256), converts to Grayscale, and generates HSV maps.
   * **Feature Extraction:** Extracts color analytics (RGB/HSV histograms), texture signatures (Sobel Gradient/Laplacian), lesion profiles, and tissue integrity metrics.
3. **Intelligence Tier (Machine Learning):** 
   * Data is normalized using `StandardScaler` (Mean Centering & Variance Unit Scaling).
   * Classification is handled by an **Ensemble Random Forest** (250 Decision Trees).
   * Confidence scores are mapped using **Isotonic Probability Calibration**.
4. **Persistence Tier:** A static metadata dictionary attaches human-readable names, descriptions, severities, and treatments to the model's matrix output.

## Model Performance
The Random Forest classifier was rigorously evaluated using Precision, Recall, and F1-scores across multiple classes. 

* **Overall Validation Accuracy:** `94.56%`
* **Macro Avg F1-Score:** `0.95`
* **Weighted Avg F1-Score:** `0.95`

The model demonstrates exceptional reliability, scoring `0.98` precision on healthy plants and maintaining `>0.90` precision across all disease categories.

## Getting Started

### Prerequisites
*(Update these based on your specific tech stack, e.g., Python 3.9+, Node.js, etc.)*
* Python 3.x
* OpenCV
* Scikit-Learn
* Flask / FastAPI (for the Application Tier)

### Installation & Run
```bash
# Clone the repository
git clone https://github.com/M-S-H-Git/LeafScan.git
cd LeafScan/LeafScan_CODE

# Install dependencies
pip install -r requirements.txt

# Run the application server
python app.py
