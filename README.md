# 🩺 Skin Cancer Detection & Diagnostics Web Application

An end-to-end machine learning web application that classifies skin lesions into 7 distinct categories. It features a Convolutional Neural Network (CNN) trained on the ISIC dataset and a modern, responsive Glassmorphism dashboard built with Flask and Chart.js for real-time diagnostics tracking and visualization.

---

## 🌟 Key Features

- **Deep Learning Classification**: Predicts 7 types of skin lesions with confidence percentage mapping and full probability distributions.
- **Glassmorphism Dashboard**: Interactive UI visualizing cumulative diagnostic statistics, total scans, and class distribution using dynamic charts.
- **Secure Authentication**: Session-based login system protecting backend routes and diagnostic data.
- **Model Evaluation Views**: Integrated metrics section showing the training history (Accuracy/Loss curves) and Confusion Matrix.
- **Simple Setup**: Includes quick-start batch scripts and pre-configured settings.

---

## 📂 Project Directory Structure

```text
Skin_Cancer_detection/
├── app.py                     # Main Flask web application backend
├── train_model.ipynb          # Jupyter Notebook for CNN model training & evaluation
├── generate_metrics.py        # Helper script for performance metrics calculation
├── run_app.bat                # Windows batch script to launch the server
├── requirements.txt           # Python library dependencies
├── stats.json                 # JSON file storing live statistics (gitignored)
├── skin_cancer_model.keras    # Trained Keras CNN model weights (gitignored)
├── static/                    # Frontend static assets
│   ├── style.css              # Glassmorphism styling rules
│   ├── training_history.png   # Epoch-by-epoch model training graphs
│   ├── confusion_matrix.png   # Performance evaluation heatmap
│   ├── metrics.json           # JSON metrics for classification report
│   └── uploads/               # Directory for user uploaded images (gitignored)
└── templates/                 # HTML templates
    ├── index.html             # Diagnostic upload page
    ├── dashboard.html         # Live telemetry & charts dashboard
    ├── about.html             # Model performance metrics page
    └── login.html             # User login portal
```

---

## 🔬 Model & Dataset Details

The classifier is a Custom Convolutional Neural Network (CNN) built with TensorFlow/Keras and trained on the **ISIC 2018** dataset.

### Target Classes (7 Types of Skin Lesions)
1. **MEL**: Melanoma
2. **NV**: Melanocytic Nevus (Common Mole)
3. **BCC**: Basal Cell Carcinoma
4. **AKIEC**: Actinic Keratosis / Bowen's Disease
5. **BKL**: Benign Keratosis (Solar Lentigines / Seborrheic Keratoses)
6. **DF**: Dermatofibroma
7. **VASC**: Vascular Lesion (Angiomas, Angiokeratomas, Pyogenic Granulomas)

### CNN Architecture
- **Input layer**: `64x64x3` RGB image normalization (`1./255`)
- **Conv2D Block 1**: 32 filters (`3x3`), ReLU activation + MaxPooling (`2x2`)
- **Conv2D Block 2**: 64 filters (`3x3`), ReLU activation + MaxPooling (`2x2`)
- **Dense Section**: Flatten + Fully Connected layer (128 nodes, ReLU)
- **Regularization**: Dropout layer (`0.5`) to prevent overfitting
- **Output layer**: Dense layer with 7 nodes, Softmax activation

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone & Set Up Directory
Open your terminal and clone/locate your project directory:
```bash
git clone <your-repository-url>
cd Skin_Cancer_detection
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Phase 1: Model Training (Optional)
If you need to train the model weights from scratch or generate performance charts:
1. Ensure the ISIC dataset files (CSV and raw images) are placed in the `archive/` or `dataset/` directory.
2. Update the paths inside `train_model.ipynb` if necessary.
3. Run the cells in `train_model.ipynb` to train the model, save `skin_cancer_model.keras`, and export the metrics to the `static/` directory.

### Phase 2: Running the Web App
Simply run the startup script (Windows) or execute via python:

**On Windows:**
Double-click `run_app.bat` or run:
```cmd
run_app.bat
```

**Using Python directly:**
```bash
python app.py
```

The Flask app will start locally at **`http://127.0.0.1:5000`**.

---

## 🔑 Demo Login Credentials

To access the diagnostic features, use the default credentials configured in `app.py`:

- **Username**: `SagarDP`
- **Password**: `PavanGC`

*(Note: These can be customized directly in the `/login` route handler inside `app.py`)*

---

## 📊 Evaluation & Diagnostics View

Once training is complete, the application pulls model telemetry directly into the frontend. You can view the following assets inside the application:
- **Accuracy & Loss History**: Track convergence across training epochs (`static/training_history.png`).
- **Confusion Matrix Heatmap**: Check inter-class performance metrics (`static/confusion_matrix.png`).
- **Precision, Recall & F1-Score**: Real-time classification report loaded from `static/metrics.json` inside the **About/Metrics** tab.

---

## ⚖️ Disclaimer
*This application is designed as an educational prototype and a machine learning demonstration. It is not intended to serve as a medical diagnostic tool or replace professional clinical consultations.*
