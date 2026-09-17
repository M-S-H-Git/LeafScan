"""
Feature extraction for plant disease detection using OpenCV and NumPy.
Extracts color histograms, texture features (LBP), and shape features.
"""

import cv2
import numpy as np
from PIL import Image
import io
import base64


def preprocess_image(image_bytes, target_size=(224, 224)):
    """Convert image bytes to numpy array and resize."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size)
    return np.array(img)


def extract_color_histogram(img, bins=32):
    """Extract color histogram features from RGB channels."""
    features = []
    for channel in range(3):
        hist = np.histogram(img[:, :, channel], bins=bins, range=(0, 256))[0]
        hist = hist.astype(float) / (hist.sum() + 1e-8)
        features.extend(hist.tolist())
    return features


def extract_hsv_features(img, bins=32):
    """Extract HSV color space features — great for detecting yellowing, browning."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    features = []
    for channel in range(3):
        hist = np.histogram(hsv[:, :, channel], bins=bins, range=(0, 256))[0]
        hist = hist.astype(float) / (hist.sum() + 1e-8)
        features.extend(hist.tolist())
    return features


def extract_texture_features(img):
    """Extract texture features using gradient-based statistics."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Sobel gradients
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)

    features = [
        float(np.mean(magnitude)),
        float(np.std(magnitude)),
        float(np.percentile(magnitude, 75)),
        float(np.percentile(magnitude, 95)),
    ]

    # Laplacian (sharpness / lesion edges)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    features += [float(np.var(lap)), float(np.mean(np.abs(lap)))]

    return features


def extract_lesion_features(img):
    """Detect and quantify lesion-like regions (spots, patches)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Brown/yellow lesion mask
    lower_brown = np.array([10, 40, 40])
    upper_brown = np.array([30, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

    # Dark spot mask
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 60])
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)

    # Yellow mask
    lower_yellow = np.array([20, 60, 60])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    total_px = img.shape[0] * img.shape[1]
    brown_ratio = float(np.sum(brown_mask > 0)) / total_px
    dark_ratio = float(np.sum(dark_mask > 0)) / total_px
    yellow_ratio = float(np.sum(yellow_mask > 0)) / total_px

    # Contour count (number of lesion regions)
    combined = cv2.bitwise_or(brown_mask, dark_mask)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lesion_count = len([c for c in contours if cv2.contourArea(c) > 30])

    return [brown_ratio, dark_ratio, yellow_ratio, float(lesion_count) / 100.0]


def extract_green_health_features(img):
    """Measure overall greenness/health of the leaf."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Healthy green range
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    total_px = img.shape[0] * img.shape[1]
    green_ratio = float(np.sum(green_mask > 0)) / total_px

    # Mean saturation and value in green areas
    green_pixels = hsv[green_mask > 0]
    if len(green_pixels) > 0:
        mean_sat = float(np.mean(green_pixels[:, 1])) / 255.0
        mean_val = float(np.mean(green_pixels[:, 2])) / 255.0
    else:
        mean_sat, mean_val = 0.0, 0.0

    return [green_ratio, mean_sat, mean_val]


def extract_all_features(image_bytes):
    """Extract complete feature vector from image bytes."""
    img = preprocess_image(image_bytes)

    features = []
    features.extend(extract_color_histogram(img))
    features.extend(extract_hsv_features(img))
    features.extend(extract_texture_features(img))
    features.extend(extract_lesion_features(img))
    features.extend(extract_green_health_features(img))

    return np.array(features, dtype=np.float32)


def image_to_base64(image_bytes):
    """Convert image bytes to base64 string for frontend display."""
    return base64.b64encode(image_bytes).decode("utf-8")


def annotate_image(image_bytes):
    """Draw lesion contours on the image for visualization."""
    img = preprocess_image(image_bytes)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower_brown = np.array([10, 40, 40])
    upper_brown = np.array([30, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 60])
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)

    combined = cv2.bitwise_or(brown_mask, dark_mask)
    kernel = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = img.copy()
    for c in contours:
        if cv2.contourArea(c) > 50:
            cv2.drawContours(annotated, [c], -1, (255, 80, 80), 2)

    pil_img = Image.fromarray(annotated)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()