
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = os.path.join("leukemia_classifier_model.h5")
model = load_model(MODEL_PATH)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    prediction = model.predict(img_array)
    return int(np.argmax(prediction))

@app.route("/")
def home():
    return "Leukemia Model API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    filepath = "temp.jpg"
    file.save(filepath)
    result = predict_image(filepath)
    return jsonify({"prediction": result})
with open("app.py", "w") as f:
    f.write(code)
