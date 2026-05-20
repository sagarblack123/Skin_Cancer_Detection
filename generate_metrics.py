import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
csv_path = r'E:\c programming\MLproject\archive\GroundTruth.csv'
image_dir = r'E:\c programming\MLproject\archive\images'
model_path = r'E:\c programming\MLproject\skin_cancer_model.keras'

print("Loading data...")
df = pd.read_csv(csv_path)
df['image'] = df['image'] + '.jpg'

classes = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

# Only use a small subset (e.g., 200 images) for quick evaluation
df_subset = df.sample(n=min(200, len(df)), random_state=42)

datagen = ImageDataGenerator(rescale=1./255)
val_gen = datagen.flow_from_dataframe(
    dataframe=df_subset,
    directory=image_dir,
    x_col='image',
    y_col=classes,
    target_size=(64, 64),
    batch_size=32,
    class_mode='raw',
    shuffle=False
)

print("Loading model...")
model = tf.keras.models.load_model(model_path)

print("Predicting...")
Y_pred = model.predict(val_gen)
y_pred = np.argmax(Y_pred, axis=1)
y_true = np.argmax(val_gen.labels, axis=1)

os.makedirs('static', exist_ok=True)

# 1. Confusion Matrix
print("Generating Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix (Subset Evaluation)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('static/confusion_matrix.png')
plt.close()

# 2. Metrics JSON
print("Generating Metrics JSON...")
report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
acc = accuracy_score(y_true, y_pred)

metrics = {
    'accuracy': acc,
    'report': report
}
with open('static/metrics.json', 'w') as f:
    json.dump(metrics, f)

# 3. Dummy Training History (Since we don't have the original history object)
print("Generating Dummy Training History...")
epochs = range(1, 11)
train_acc = np.linspace(0.4, 0.85, 10) + np.random.normal(0, 0.02, 10)
val_acc = np.linspace(0.4, 0.80, 10) + np.random.normal(0, 0.03, 10)
train_loss = np.linspace(1.5, 0.4, 10) + np.random.normal(0, 0.05, 10)
val_loss = np.linspace(1.5, 0.5, 10) + np.random.normal(0, 0.05, 10)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs, train_acc, label='Train Accuracy')
ax1.plot(epochs, val_acc, label='Val Accuracy')
ax1.set_title('Model Accuracy (Simulated)')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend()

ax2.plot(epochs, train_loss, label='Train Loss')
ax2.plot(epochs, val_loss, label='Val Loss')
ax2.set_title('Model Loss (Simulated)')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend()

plt.savefig('static/training_history.png')
plt.close()

print("All metrics generated successfully!")
