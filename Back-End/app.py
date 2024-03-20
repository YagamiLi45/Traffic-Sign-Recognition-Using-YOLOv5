from flask import Flask, render_template, request, jsonify
from testing import YOLO_Pred
import cv2
import numpy as np
import base64
from flask_cors import CORS

app = Flask(__name__, template_folder='../Front-End', static_folder='../Front-End')

# Load the YOLO model
onnx_model_path = r'C:\All Projects\Final Year Project\Model77\weights\best.onnx'
data_yaml_path = r'C:\All Projects\Final Year Project\data123.yaml'
yolo_pred = YOLO_Pred(onnx_model_path, data_yaml_path)

CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_page')
def upload_page():
    return render_template('UploadData.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the file from the request
        file = request.files['file']

        # Read the image file
        image_np = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Perform traffic sign detection and recognition
        result_image = yolo_pred.predictions(image_np)

        # Convert the result image to base64 for displaying in HTML
        _, buffer = cv2.imencode('.jpg', result_image)
        result_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'result_image': result_image_base64})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

