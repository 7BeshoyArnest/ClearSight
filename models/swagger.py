from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
import os
from flasgger import Swagger

# Initialize Flask app
app = Flask(__name__)
swagger = Swagger(app)

# Load model once on startup
MODEL_PATH = r'C:\Users\AL-MASA\Desktop\GitHub_Models\Model2\model.h5'
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
disease_names = ['cataract', 'diabetic Retinopathy', 'glaucoma', 'normal']

# Image preprocessing function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """
    File Upload Endpoint
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: image
        in: formData
        type: file
        required: true
    responses:
      200:
        description: Prediction response
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Save file temporarily
    filepath = os.path.join('temp', file.filename)
    os.makedirs('temp', exist_ok=True)
    file.save(filepath)

    try:
        input_data = preprocess_image(filepath)
        predictions = model.predict(input_data)
        predicted_class = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        if confidence < 0.6:
            result = {'Prediction': 'Unknown disease', 'Confidence': confidence}
        else:
            result = {
                'Prediction': disease_names[predicted_class],
                'Confidence': confidence
            }
        return jsonify(result)

    finally:
        # Clean up temporary file
        os.remove(filepath)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
