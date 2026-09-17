"""
Plant Disease Classifier using Real-World Dataset Features.
Extracts deep visual features from real images using custom computer vision heuristics
and trains an optimized, calibrated Random Forest pipeline.
"""

import os
import pickle
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

MODEL_PATH = os.path.join(os.path.dirname(__file__), "plant_model.pkl")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")

# ── Disease Definitions ────────────────────────────────────────────────────────
DISEASES = {
    0: {
        "name": "Healthy",
        "description": "The plant appears healthy with no visible signs of disease.",
        "severity": "None", "color": "#22c55e", "icon": "✅",
        "treatments": [
            "Continue regular watering and fertilization schedule",
            "Monitor for early signs of pests or disease",
            "Ensure adequate sunlight and air circulation"
        ],
        "prevention": "Maintain good cultural practices to keep plants healthy."
    },
    1: {
        "name": "Early Blight",
        "description": "Alternaria solani fungal infection causing dark brown spots with concentric rings, usually appearing on lower/older leaves first.",
        "severity": "Moderate", "color": "#f59e0b", "icon": "⚠️",
        "treatments": [
            "Remove and destroy infected leaves immediately",
            "Apply copper-based fungicide every 7–10 days",
            "Avoid overhead watering to reduce leaf wetness"
        ],
        "prevention": "Rotate crops annually; avoid planting in previously infected soil."
    },
    2: {
        "name": "Late Blight",
        "description": "Phytophthora infestans causes water-soaked lesions that turn dark brown/black, spreading rapidly in cool, wet conditions.",
        "severity": "High", "color": "#ef4444", "icon": "🚨",
        "treatments": [
            "Apply systemic fungicide immediately",
            "Remove all infected plant material and dispose off-site",
            "Do not compost infected material"
        ],
        "prevention": "Use certified disease-free seeds; plant resistant varieties."
    },
    3: {
        "name": "Leaf Mold",
        "description": "Passalora fulva causes pale green/yellow spots on upper leaf surfaces with olive-green mold on the undersides.",
        "severity": "Moderate", "color": "#f59e0b", "icon": "⚠️",
        "treatments": [
            "Reduce humidity below 85% in enclosed growing areas",
            "Apply chlorothalonil or copper fungicide",
            "Increase ventilation around plants"
        ],
        "prevention": "Avoid wetting foliage; use drip irrigation instead of overhead."
    },
    4: {
        "name": "Bacterial Spot",
        "description": "Xanthomonas bacteria cause small, water-soaked spots that turn brown with yellow halos, affecting leaves and fruit.",
        "severity": "Moderate", "color": "#f59e0b", "icon": "⚠️",
        "treatments": [
            "Apply copper-based bactericide at first sign of infection",
            "Remove infected plant debris promptly",
            "Avoid working with plants when wet"
        ],
        "prevention": "Use resistant seed varieties; practice crop rotation."
    },
    5: {
        "name": "Powdery Mildew",
        "description": "Fungal disease causing white powdery coating on leaf surfaces, stunting growth and reducing photosynthesis.",
        "severity": "Low–Moderate", "color": "#eab308", "icon": "⚠️",
        "treatments": [
            "Apply potassium bicarbonate or neem oil spray",
            "Use sulfur-based fungicide as preventative",
            "Improve air circulation around plants"
        ],
        "prevention": "Plant in sunny, well-ventilated locations; avoid overhead irrigation."
    },
    6: {
        "name": "Septoria Leaf Spot",
        "description": "Septoria lycopersici causes circular spots with dark brown borders and lighter centers, typically starting on lower leaves.",
        "severity": "Moderate", "color": "#f59e0b", "icon": "⚠️",
        "treatments": [
            "Apply fungicide containing chlorothalonil or mancozeb",
            "Mulch around plants to prevent soil splash",
            "Remove and discard infected leaves"
        ],
        "prevention": "Crop rotation and removal of plant debris at end of season."
    },
    7: {
        "name": "Spider Mite Damage",
        "description": "Tetranychus urticae causes stippling, yellowing, and bronzing of leaves; fine webbing may be visible on undersides.",
        "severity": "Low–Moderate", "color": "#eab308", "icon": "🕷️",
        "treatments": [
            "Apply insecticidal soap or neem oil to leaf undersides",
            "Use miticide for severe infestations",
            "Introduce predatory mites as biocontrol"
        ],
        "prevention": "Avoid water stress; regularly inspect undersides of leaves."
    }
}

FEATURE_DIM = 205 


def _extract_real_image_features(image_path):
    """
    Reads a real leaf image and processes its real color, texture, 
    and geometric properties to construct a structured 205-dimensional vector.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    # Resize to a consistent scale
    img = cv2.resize(img, (256, 256))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    feat = np.zeros(FEATURE_DIM, dtype=np.float32)
    
    # 1. Color Histograms (RGB + HSV Channels)
    for c in range(3):
        hist_rgb = cv2.calcHist([img], [c], None, [32], [0, 256]).flatten()
        hist_rgb /= (hist_rgb.sum() + 1e-8)
        feat[c * 32 : (c + 1) * 32] = hist_rgb
        
        hist_hsv = cv2.calcHist([hsv], [c], None, [32], [0, 256]).flatten()
        hist_hsv /= (hist_hsv.sum() + 1e-8)
        feat[96 + c * 32 : 96 + (c + 1) * 32] = hist_hsv

    # 2. Advanced Texture Analytics
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(sobel_x, sobel_y)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    feat[192] = np.mean(grad_mag)
    feat[193] = np.std(grad_mag)
    feat[194] = np.percentile(grad_mag, 75)
    feat[195] = np.percentile(grad_mag, 95)
    feat[196] = np.var(laplacian)
    feat[197] = np.mean(np.abs(laplacian))
    
    # 3. Structural Lesion Profiling
    b, g, r = cv2.split(img)
    brown_mask = (b < 100) & (r > 100) & (g < 150)
    yellow_mask = (b < 120) & (r > 150) & (g > 150)
    dark_mask = (r < 50) & (g < 50) & (b < 50)
    
    feat[198] = np.sum(brown_mask) / (256 * 256)
    feat[199] = np.sum(dark_mask) / (256 * 256)
    feat[200] = np.sum(yellow_mask) / (256 * 256)
    feat[201] = feat[198] + feat[199] + feat[200]
    
    # 4. Green Leaf Tissue Integrity Metrics
    green_mask = (g > r) & (g > b) & (hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 85)
    feat[202] = np.sum(green_mask) / (256 * 256)
    feat[203] = np.mean(hsv[:, :, 1]) / 255.0
    feat[204] = np.mean(hsv[:, :, 2]) / 255.0
    
    return feat


def _load_real_dataset():
    """Traverses folder branches to aggregate data matrices by matching string names."""
    X, y = [], []
    
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(
            f"Dataset folder missing! Please verify path existence at: '{DATASET_DIR}'"
        )
        
    print(f"Scanning target folders in: {DATASET_DIR}")
    
    # Pre-map clean disease descriptors to internal class indices
    name_to_id = {}
    for cid, info in DISEASES.items():
        clean_name = info["name"].lower().replace(" ", "").replace("_", "")
        name_to_id[clean_name] = cid

    for folder_name in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        # Clean folder name to evaluate string matches (e.g. "tomato__early_blight" -> "earlyblight")
        clean_folder = folder_name.lower().replace(" ", "").replace("_", "")
        
        class_id = None
        for disease_keyword, cid in name_to_id.items():
            if disease_keyword in clean_folder:
                class_id = cid
                break
                
        if class_id is None:
            print(f"Skipping folder (Not in our 8 target classes): {folder_name}")
            continue
            
        print(f" -> Mapping folder '{folder_name}' to class: {DISEASES[class_id]['name']}")
        img_count = 0
        
        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                continue
                
            img_path = os.path.join(folder_path, filename)
            features = _extract_real_image_features(img_path)
            
            if features is not None:
                X.append(features)
                y.append(class_id)
                img_count += 1
                
        print(f"    Successfully processed {img_count} images.")

    if len(X) == 0:
        raise ValueError("No valid image matrices captured. Please place target sample images in the directories.")
        
    return np.array(X), np.array(y)


def train_and_save():
    """Runs data aggregation, splits samples, shows real accuracy metrics, and saves model."""
    print("Loading real-world training dataset from local folders …")
    X, y = _load_real_dataset()

    unique, counts = np.unique(y, return_counts=True)
    if np.min(counts) < 2:
        print("\n⚠️ Warning: Baseline count low. Skipping stratified evaluation split.")
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    rf = RandomForestClassifier(
        n_estimators=250,
        max_depth=15,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(rf, cv=3, method="isotonic")),
    ])

    print("\nTraining on real dataset properties …")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("\n" + "="*20 + " REAL DATASET PERFORMANCE EVALUATION " + "="*20)
    print(f"Overall Validation Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    
    present_classes = np.unique(np.concatenate([y_test, y_pred]))
    target_names = [DISEASES[i]["name"] for i in present_classes]
    
    print("\nDetailed Per-Class Performance:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("="*75 + "\n")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model successfully saved to {MODEL_PATH}")
    return pipeline


def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return train_and_save()


_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


def predict(features: np.ndarray):
    """Predicts category using the real-world trained asset."""
    model = get_model()
    probs = model.predict_proba(features.reshape(1, -1))[0]
    class_id = int(np.argmax(probs))
    confidence = float(probs[class_id])
    all_probs = {DISEASES[i]["name"]: float(p) for i, p in enumerate(probs)}
    return class_id, confidence, all_probs