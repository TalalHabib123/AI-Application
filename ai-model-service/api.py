from flask import Flask, request, jsonify
from keras.models import load_model
from keras.utils import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import preprocess_input
import numpy as np
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

diseases = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']


if tf.test.is_gpu_available():
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
else:
    strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")

with strategy.scope():
    try:
        model = load_model('model.h5')
        print("Model loaded")
    except Exception as e:
        print(f"Error loading model: {e}")


@app.route('/', methods=['GET'])
def ServerStatus():
    return 'Server is running'

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    
    file = request.files['file']
    img = Image.open(file)
    img = img.resize((256, 256))
    img = img.convert('RGB')
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    stimulate_predictions =[]
    for _ in range(10):
        predictions = model.predict(x, verbose=0)
        stimulate_predictions.append(predictions)
    predictions = np.mean(np.array(stimulate_predictions), axis=0)
    labels = np.argmax(predictions, axis=1)
    pred_class = int(labels[0])
    pred_class = diseases[pred_class]
    # Return the prediction
    return jsonify({'class': pred_class, 'label': int(labels[0]), 'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run()