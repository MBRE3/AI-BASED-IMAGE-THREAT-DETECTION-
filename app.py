from flask import Flask, request, jsonify
import numpy as np
from PIL import Image, ExifTags
import cv2
import io
import csv
import os
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)

# CSV file to store results
DATASET_FILE = "learn_dataset.csv"

# ---------- Rule Functions ----------

def check_exif(image):
    try:
        exif = image._getexif()
        if not exif:
            return 30
        exif_data = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        essential = ["DateTime", "Make", "Model"]
        missing = [tag for tag in essential if tag not in exif_data]
        return len(missing) * 10
    except Exception:
        return 30

def jpeg_artifact_score(np_img):
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    block_scores = []
    for i in range(0, h - 8, 8):
        for j in range(0, w - 8, 8):
            block = gray[i:i+8, j:j+8].astype(np.float32)
            dct = cv2.dct(block)
            high_freq_energy = np.sum(np.abs(dct[4:, 4:]))
            block_scores.append(high_freq_energy)
    avg_energy = np.mean(block_scores) if block_scores else 0
    score = 100 if avg_energy > 2500 else (avg_energy / 2500) * 100
    return score

def color_distribution_score(np_img):
    score = 0
    for channel in range(3):
        hist = cv2.calcHist([np_img], [channel], None, [256], [0, 256])
        peaks = np.sum(hist > 1000)
        if peaks > 150:
            score += 20
    return score

def edge_score(np_img):
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size * 100
    if edge_density < 5 or edge_density > 15:
        return 20
    return 0

def clone_score(np_img):
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    patch = gray[h//4:h//4 + 30, w//4:w//4 + 30]
    res = cv2.matchTemplate(gray, patch, cv2.TM_CCOEFF_NORMED)
    matches = np.sum(res > 0.95)
    return 30 if matches > 5 else 0

def splicing_score(np_img):
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    h, w = gradient_mag.shape
    block_h, block_w = 32, 32
    blocks = []

    for y in range(0, h - block_h, block_h):
        for x in range(0, w - block_w, block_w):
            block = gradient_mag[y:y+block_h, x:x+block_w]
            block_var = np.var(block)
            blocks.append(block_var)

    if len(blocks) == 0:
        return 0

    global_var = np.var(blocks)
    return min(global_var * 2, 40)

# ---------- Visualization ----------

def generate_splicing_heatmap(np_img):
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    h, w = gradient_mag.shape
    heatmap = np.zeros((h, w), dtype=np.float32)

    block_h, block_w = 32, 32
    for y in range(0, h - block_h, block_h):
        for x in range(0, w - block_w, block_w):
            block = gradient_mag[y:y+block_h, x:x+block_w]
            block_var = np.var(block)
            heatmap[y:y+block_h, x:x+block_w] = block_var

    norm_heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_map = cv2.applyColorMap(norm_heatmap, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(np_img, 0.6, color_map, 0.4, 0)

    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')

# ---------- Helper Functions ----------

def determine_threat_types(scores):
    threats = []
    if scores['exif'] >= 20:
        threats.append("Metadata issues")
    if scores['jpeg_artifacts'] >= 50:
        threats.append("Compression artifacts")
    if scores['color_distribution'] >= 20:
        threats.append("Unnatural color distribution")
    if scores['edges'] > 0:
        threats.append("Irregular edge patterns")
    if scores['clones'] > 0:
        threats.append("Clone regions detected")
    if scores.get('splicing', 0) >= 25:
        threats.append("Possible image splicing")
    return threats if threats else ["None"]

def append_to_dataset(scores, total_score, threat_detected, threat_types):
    file_exists = os.path.isfile(DATASET_FILE)
    with open(DATASET_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["exif", "jpeg_artifacts", "color_distribution", "edges", "clones", "splicing",
                             "total_score", "threat_detected", "threat_types"])
        writer.writerow([
            float(scores['exif']),
            float(scores['jpeg_artifacts']),
            float(scores['color_distribution']),
            float(scores['edges']),
            float(scores['clones']),
            float(scores['splicing']),
            float(total_score),
            bool(threat_detected),
            "; ".join(threat_types)
        ])

# ---------- Master Analysis Function ----------

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Invalid image file: {str(e)}'}), 400

    np_img = np.array(image.resize((256, 256)))

    scores = {
        'exif': check_exif(image),
        'jpeg_artifacts': jpeg_artifact_score(np_img),
        'color_distribution': color_distribution_score(np_img),
        'edges': edge_score(np_img),
        'clones': clone_score(np_img),
        'splicing': splicing_score(np_img)
    }

    total_score = sum(scores.values())
    threat_types = determine_threat_types(scores)
    threat_detected = (threat_types != ["None"]) or (total_score >= 50)
    confidence = min((total_score / 100) * 100, 100)

    append_to_dataset(scores, total_score, threat_detected, threat_types)

    splicing_visual = generate_splicing_heatmap(np_img)

    result = {
        'threat_detected': bool(threat_detected),
        'confidence': float(round(confidence, 2)),
        'total_score': float(round(total_score, 2)),
        'threat_types': threat_types,
        'individual_scores': {k: float(v) for k, v in scores.items()},
        'splicing_visual': splicing_visual
    }
    return jsonify(result), 200

if __name__ == '__main__':
    app.run(debug=True)
