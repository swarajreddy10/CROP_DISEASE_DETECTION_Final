from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load trained model
MODEL_PATH = "trained_plant_disease_model11.keras"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define class labels
CLASS_LABELS = [
    "Apple Scab", "Black Rot", "Cedar Apple Rust", "Healthy Apple",
    "Healthy Blueberry", "Cherry Powdery Mildew", "Healthy Cherry",
    "Cercospora Leaf Spot", "Common Rust", "Northern Leaf Blight", "Healthy Corn",
    "Black Rot", "Esca (Black Measles)", "Leaf Blight", "Healthy Grape",
    "Haunglongbing (Citrus Greening)", "Bacterial Spot", "Healthy Peach",
    "Bacterial Spot", "Healthy Pepper", "Early Blight", "Late Blight", "Healthy Potato",
    "Healthy Raspberry", "Healthy Soybean", "Squash Powdery Mildew",
    "Strawberry Leaf Scorch", "Healthy Strawberry",
    "Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold", "Septoria Leaf Spot",
    "Spider Mites", "Target Spot", "Yellow Leaf Curl Virus", "Mosaic Virus", "Healthy Tomato"
]

# Define upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(img_path):
    # Changed from 256x256 to 128x128 to match the model's expected input shape
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model failed to load. Please check server logs.'}), 500
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Save the uploaded image
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Preprocess the image
        img_array = preprocess_image(file_path)
        
        # Predict
        predictions = model.predict(img_array)[0]
        predicted_index = np.argmax(predictions)
        confidence = float(predictions[predicted_index] * 100)
        predicted_class = CLASS_LABELS[predicted_index]
        
        return jsonify({'class': predicted_class, 'confidence': confidence})
    except Exception as e:
        # Always return JSON for any errors
        return jsonify({'error': str(e)}), 500

# Add an error handler for all exceptions to ensure JSON responses
@app.errorhandler(Exception)
def handle_exception(e):
    # Return JSON instead of HTML for any errors
    return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)