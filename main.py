from flask import Flask, request, jsonify, render_template, session
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tensorflow as tf
import logging
import base64
from flask_session import Session
from collections import deque

# Logging setup
logging.basicConfig(level=logging.DEBUG)

# Model class names and settings
scvd_classes = ['Normal', 'Violence']
IMG_SIZE = 128
FRAME_COUNT = 15
QUEUE_SIZE = 5  # Queue size for smoothing
CONFIDENCE_THRESHOLD = 0.7  # Confidence threshold for predictions

app = Flask(__name__)
CLASS_NAMES = ["Non-Violence", "Violence"]  # Model's classes
app.config['SECRET_KEY'] = '234df546434'  # Random secret key
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Load pre-trained models
model = tf.keras.models.load_model('bestt.keras')
model2 = tf.keras.models.load_model('cctvmodel.keras')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to load video file, preprocess frames
def load_video(path, nframes=FRAME_COUNT, size=(IMG_SIZE, IMG_SIZE), skip_frames=1):
    frames = []
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {path}")
    for _ in range(nframes):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        frame = frame[:, :, [2, 1, 0]]  # BGR to RGB
        frame = frame / 255.0
        frames.append(frame)
        for _ in range(skip_frames):
            cap.grab()
    cap.release()
    return np.array(frames)  # Ensure that frames are converted to a NumPy array

# Allowed file check
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route: Home page
@app.route('/')
def home():
    return render_template('service.html')

# Route: Index page
@app.route('/index')
def index():
    return render_template('index.html')

# Route: Video file classification
@app.route('/classify_video', methods=['POST'])
def classify_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']

    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if video_file and allowed_file(video_file.filename):
        filename = secure_filename(video_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(filepath)

        try:
            logging.info(f"Processing video: {filepath}")
            frames = load_video(filepath)
            logging.info(f"Loaded {len(frames)} frames")

            # Ensure we have the correct number of frames
            if len(frames) < FRAME_COUNT:
                logging.warning(f"Not enough frames: {len(frames)}. Padding...")
                frames = np.pad(frames, ((0, FRAME_COUNT - len(frames)), (0, 0), (0, 0), (0, 0)), mode='edge')
            elif len(frames) > FRAME_COUNT:
                logging.warning(f"Too many frames: {len(frames)}. Truncating...")
                frames = frames[:FRAME_COUNT]

            frames_array = np.expand_dims(frames, axis=0)  # Add batch dimension
            logging.info(f"Input shape: {frames_array.shape}")

            # Make prediction
            prediction = model2.predict(frames_array)
            logging.info(f"Raw prediction: {prediction}")

            # For binary classification, assuming single output neuron:
            predicted_class = (prediction > 0.5).astype(int)[0][0]  # 0 or 1
            predicted_label = scvd_classes[predicted_class]

            logging.info(f"Predicted class: {predicted_class}, label: {predicted_label}")

            os.remove(filepath)
            return jsonify({'result': predicted_label})

        except Exception as e:
            logging.error(f"Error processing video: {str(e)}")
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

# Route: Video stream classification
@app.route('/classify_video_stream', methods=['POST'])
def classify_video_stream():
    try:
        # Get the base64 encoded image from the request
        image_data = request.json['image']

        # Decode the base64 image
        image_data = base64.b64decode(image_data.split(',')[1])

        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)

        # Decode image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Initialize or get the frame buffer from the session
        if 'frame_buffer' not in session:
            session['frame_buffer'] = []
        frame_buffer = session['frame_buffer']

        # Preprocess the frame
        processed_frame = preprocess_frame(frame)

        # Add the processed frame to the buffer
        frame_buffer.append(processed_frame)

        # Keep only the last 15 frames
        frame_buffer = frame_buffer[-15:]

        # Initialize or get the prediction queue from the session
        if 'prediction_queue' not in session:
            session['prediction_queue'] = []
        prediction_queue = session['prediction_queue']

        predicted_label = "Collecting frames"
        confidence = 0.0

        # Only make a prediction if we have 15 frames
        if len(frame_buffer) == 15:
            # Prepare input for the model
            model_input = np.expand_dims(np.array(frame_buffer), axis=0)

            # Make prediction
            prediction = model.predict(model_input)

            print("1. Raw prediction shape:", prediction.shape)
            print("2. Raw prediction:", prediction)

            # Ensure prediction is 1D
            prediction = prediction.flatten()

            print("3. Flattened prediction:", prediction)

            # Smooth predictions
            smoothed_prediction = smooth_predictions(prediction, prediction_queue)

            print("4. Smoothed prediction:", smoothed_prediction)

            # Get the predicted class and confidence
            predicted_class = 1 if smoothed_prediction[0] > 0.5 else 0
            confidence = smoothed_prediction[0] if predicted_class == 1 else 1 - smoothed_prediction[0]

            print("5. Predicted class:", predicted_class)
            print("6. Confidence:", confidence)

            # Apply a lower confidence threshold
            if confidence > 0.55:  # Lowered from 0.7
                predicted_label = CLASS_NAMES[predicted_class]
            else:
                predicted_label = "Uncertain"

            # Apply majority voting
            if len(prediction_queue) == QUEUE_SIZE:
                is_violence = majority_voting(prediction_queue)
                predicted_label = CLASS_NAMES[1] if is_violence else CLASS_NAMES[0]

        # Store the updated buffers in the session
        session['frame_buffer'] = frame_buffer
        session['prediction_queue'] = prediction_queue
        session.modified = True

        return jsonify({
            'result': predicted_label,
            'confidence': float(confidence)
        })

    except Exception as e:
        logging.error(f"Error processing video stream: {str(e)}")
        return jsonify({'error': str(e)}), 500

def preprocess_frame(frame, size=(IMG_SIZE, IMG_SIZE)):
    frame = cv2.resize(frame, size)
    frame = frame[:, :, [2, 1, 0]]  # Convert BGR to RGB
    frame = frame / 255.0  # Normalize pixel values
    return frame


def smooth_predictions(prediction, queue):
    # Ensure prediction is 1D
    prediction = prediction.flatten()

    # Add the new prediction to the queue
    queue.append(prediction[0])  # Assuming binary classification, take the first (and only) value

    # Keep only the last QUEUE_SIZE predictions
    queue = queue[-QUEUE_SIZE:]

    # Calculate the mean of the predictions in the queue
    return [np.mean(queue)]



def majority_voting(queue, threshold=0.5):  # Lowered from 0.7
    count_violence = sum(1 for pred in queue if pred > 0.5)
    return (count_violence / len(queue)) > threshold

# Route: Stream page
@app.route('/stream')
def stream():
    return render_template('about.html')

# Main function
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
