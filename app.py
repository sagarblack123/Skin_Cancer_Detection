import os
import json
from functools import wraps
from flask import Flask, request, render_template, jsonify, session, redirect, url_for
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)
app.secret_key = 'skin_cancer_detection_secret_key'

# Ensure upload directory exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model (Ensure you've run train_model.ipynb first)
MODEL_PATH = 'skin_cancer_model.keras'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Warning: Could not load model from {MODEL_PATH}. Error: {e}")
    print("Please make sure to run the Jupyter Notebook to train the model first.")
    model = None

# Class labels (Must match the ones in train_model.ipynb exactly)
CLASSES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

# Detailed names for UI display
CLASS_NAMES = {
    'MEL': 'Melanoma',
    'NV': 'Melanocytic Nevus',
    'BCC': 'Basal Cell Carcinoma',
    'AKIEC': 'Actinic Keratosis',
    'BKL': 'Benign Keratosis',
    'DF': 'Dermatofibroma',
    'VASC': 'Vascular Lesion'
}

STATS_FILE = 'stats.json'

def load_stats():
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading stats: {e}")
    return {
        'total_detections': 0,
        'class_distribution': {
            'Melanoma': 0,
            'Melanocytic Nevus': 0,
            'Basal Cell Carcinoma': 0,
            'Actinic Keratosis': 0,
            'Benign Keratosis': 0,
            'Dermatofibroma': 0,
            'Vascular Lesion': 0
        }
    }

def save_stats(stats):
    try:
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=4)
    except Exception as e:
        print(f"Error saving stats: {e}")

# Login decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    # Resize to match model's expected input
    img = cv2.resize(img, (64, 64))
    # Convert from BGR (OpenCV default) to RGB (what Keras model typically expects)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize pixel values
    img = img / 255.0
    # Expand dimensions to create batch of 1
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('logged_in'):
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        # Demo Credentials: SagarDP / PavanGC
        if username == 'SagarDP' and password == 'PavanGC':
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/stats')
@login_required
def get_stats():
    return jsonify(load_stats())

@app.route('/', methods=['GET'])
@login_required
def index():
    return render_template('index.html')

@app.route('/about', methods=['GET'])
@login_required
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading.'}), 400

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess and Predict
        processed_image = preprocess_image(filepath)
        if processed_image is None:
             return jsonify({'error': 'Error processing the image file.'}), 400
              
        predictions = model.predict(processed_image)[0]
        
        predicted_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_idx])
        predicted_class_code = CLASSES[predicted_idx]
        predicted_class_name = CLASS_NAMES.get(predicted_class_code, predicted_class_code)

        # Create dictionary of all class probabilities
        probabilities = {}
        for i, code in enumerate(CLASSES):
            probabilities[CLASS_NAMES.get(code, code)] = float(predictions[i])

        # Update cumulative stats
        stats = load_stats()
        stats['total_detections'] += 1
        if predicted_class_name in stats['class_distribution']:
            stats['class_distribution'][predicted_class_name] += 1
        else:
            stats['class_distribution'][predicted_class_name] = 1
        save_stats(stats)

        # Return file path relative to static folder for UI display
        relative_filepath = filepath.replace('\\', '/')
        
        return jsonify({
            'prediction': predicted_class_name,
            'confidence': f"{confidence * 100:.2f}%",
            'probabilities': probabilities,
            'image_path': relative_filepath
        })

@app.route('/metrics', methods=['GET'])
@login_required
def get_metrics():
    import json
    metrics_path = os.path.join('static', 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return jsonify(json.load(f))
    else:
        return jsonify({'error': 'Metrics not generated yet. Please run the notebook.'}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)
