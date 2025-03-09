from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import os
import cv2
import numpy as np

app = Flask(__name__)

# Load the trained YOLO model
model = YOLO('runs/waste_detection/best.pt')

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Load the image
    image = cv2.imread(file_path)

    # Perform prediction
    results = model(image)

    # Extract the results
    predictions = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box
            label = model.names[int(class_id)]
            predictions.append({
                'label': label,
                'confidence': float(confidence),
                'bbox': [int(x1), int(y1), int(x2), int(y2)]
            })

    # Return the predictions
    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True)