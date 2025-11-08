from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import sounddevice as sd
from extract_features import extract_visual_features, extract_audio_features
from rule_based import rule_based_scores
import joblib
import os
import json
import base64

app = Flask(__name__)
CORS(app)

# Load models with error handling
models_path = '../models'
models_path = os.path.abspath(models_path)

# Load feature names for proper ordering
feature_names = None
try:
    with open(os.path.join(models_path, 'feature_names.json'), 'r') as f:
        feature_names = json.load(f)
    print(f"Loaded {len(feature_names)} feature names")
except FileNotFoundError:
    print("Warning: feature_names.json not found. Using unsorted keys.")

try:
    svm_no = joblib.load(os.path.join(models_path, 'svm_no_glasses.pkl'))
    scaler_no = joblib.load(os.path.join(models_path, 'scaler_no.pkl'))
except FileNotFoundError:
    print("Warning: No-glasses models not found. Training required.")
    svm_no = None
    scaler_no = None

try:
    svm_glasses = joblib.load(os.path.join(models_path, 'svm_glasses.pkl'))
    scaler_glasses = joblib.load(os.path.join(models_path, 'scaler_glasses.pkl'))
except FileNotFoundError:
    print("Warning: Glasses models not found. Training required.")
    svm_glasses = None
    scaler_glasses = None

try:
    glasses_svm = joblib.load(os.path.join(models_path, 'glasses_svm.pkl'))
except FileNotFoundError:
    print("Warning: Glasses detection model not found.")
    glasses_svm = None

cap = None
stream = None
prev_frame = None
visual_window = []
audio_window = []
audio_data = []
latest_frame = None  # Store frame from frontend

@app.route('/start', methods=['POST'])
def start():
    global cap, stream, prev_frame, visual_window, audio_window, audio_data
    # Don't use cv2.VideoCapture - browser handles camera
    cap = type('DummyCap', (), {'isOpened': lambda: True})()  # Dummy object to pass checks
    def audio_callback(indata, frames, time, status):
        audio_data.append(indata.copy())
    
    stream = sd.InputStream(callback=audio_callback, samplerate=16000, channels=1)
    stream.start()
    visual_window = []
    audio_window = []
    return jsonify({'status': 'started'})

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global latest_frame
    try:
        data = request.get_json()
        if 'frame' in data:
            # Decode base64 frame
            frame_data = data['frame'].split(',')[1]  # Remove data:image/jpeg;base64, prefix
            frame_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                latest_frame = frame
                print(f"Received frame from frontend: shape={frame.shape}")
                return jsonify({'status': 'frame_received'})
        return jsonify({'error': 'Invalid frame data'})
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)})

@app.route('/process', methods=['GET'])
def process():
    global prev_frame, visual_window, audio_window, latest_frame
    try:
        if latest_frame is None:
            print("ERROR: No frame received from frontend yet")
            return jsonify({'error': 'No frame', 'prediction': 'Neutral', 'scores': {}})
        
        frame = latest_frame.copy()
        print(f"Processing frame: shape={frame.shape}")
        visual_features, visual_window = extract_visual_features(frame, prev_frame, visual_window)
        print(f"Visual features extracted: {len(visual_features)} features")
        # Log key hand gesture features for debugging
        hand_features_log = {k: v for k, v in visual_features.items() if 'hand' in k.lower() or 'finger' in k.lower() or 'convexity' in k.lower()}
        if hand_features_log:
            print(f"Hand gesture features: {hand_features_log}")
        
        audio_chunk = np.concatenate(audio_data, axis=0).flatten() if audio_data else np.zeros(16000)
        audio_data.clear()
        audio_features, audio_window = extract_audio_features(audio_chunk, sr=16000, audio_window=audio_window)
        print(f"Audio features extracted: {len(audio_features)} features")
        
        features = {**visual_features, **audio_features}
        glasses = features.get('glasses', False)
        print(f"Total features: {len(features)}, Glasses: {glasses}")
        
        # Check if models are available
        if glasses and (svm_glasses is None or scaler_glasses is None):
            prediction = 2  # Default to Neutral if models not available
        elif not glasses and (svm_no is None or scaler_no is None):
            prediction = 2  # Default to Neutral if models not available
        else:
            # Use feature_names.json for proper feature ordering (matching training)
            if feature_names is not None:
                keys = feature_names
                # Fill missing features with 0
                vector = np.array([features.get(k, 0) for k in keys]).reshape(1, -1)
            else:
                # Fallback to sorted keys if feature_names.json not available
                keys = sorted(features.keys())
                vector = np.array([features.get(k, 0) for k in keys]).reshape(1, -1)
            
            scaler = scaler_glasses if glasses else scaler_no
            svm = svm_glasses if glasses else svm_no
            
            try:
                vector = scaler.transform(vector)
                prediction = int(svm.predict(vector)[0])  # Ensure it's a Python int
                # Get prediction confidence
                proba = svm.predict_proba(vector)[0] if hasattr(svm, 'predict_proba') else None
                print(f"Prediction successful: {prediction} ({['Confident', 'Nervous', 'Neutral'][prediction]})")
                if proba is not None:
                    print(f"Confidence: {proba}")
            except Exception as e:
                print(f"Prediction error: {e}")
                print(f"Vector shape: {vector.shape}, Expected: {(1, scaler.n_features_in_)}")
                print(f"Feature keys count: {len(keys)}, First few: {keys[:5] if len(keys) >= 5 else keys}")
                prediction = 2  # Default to Neutral on error
        
        scores = rule_based_scores(features)
        print(f"Rule-based scores calculated: {scores}")
        
        # Rule-based override: If overall performance is excellent, override to Confident
        # This helps users get "Confident" when they're actually performing well
        overall_performance = np.mean(list(scores.values()))
        
        # Override prediction based on high performance scores (lenient thresholds)
        if overall_performance >= 70:
            # If overall score is 70%+, check individual metrics
            eye_ok = scores.get('eye_contact', 0) >= 65
            head_ok = scores.get('head_stability', 0) >= 65
            voice_ok = scores.get('voice', 0) >= 65
            gestures_ok = scores.get('gesture_activity', 0) >= 50
            
            # If 2+ out of 4 key metrics are good, or overall is very high, override to Confident
            good_metrics = sum([eye_ok, head_ok, voice_ok, gestures_ok])
            if good_metrics >= 2 or overall_performance >= 75:
                prediction = 0  # Override to Confident
                print(f"OVERRIDE: High performance (avg={overall_performance:.1f}%, good_metrics={good_metrics}/4) → Confident")
            elif overall_performance >= 80:
                prediction = 0  # Very high overall → Always Confident
                print(f"OVERRIDE: Very high overall score ({overall_performance:.1f}%) → Confident")
        elif overall_performance < 45:
            # Very low performance → likely Nervous
            prediction = 1
            print(f"OVERRIDE: Low performance ({overall_performance:.1f}%) → Nervous")
        
        # Ensure all values are JSON serializable
        scores_serializable = {k: float(v) if isinstance(v, (np.integer, np.floating)) else int(v) if isinstance(v, np.integer) else v 
                               for k, v in scores.items()}
        
        prev_frame = frame.copy()
        result = {'prediction': ['Confident', 'Nervous', 'Neutral'][prediction], 'scores': scores_serializable}
        print(f"Returning result: prediction={result['prediction']}, scores={result['scores']}")
        return jsonify(result)
    except Exception as e:
        print(f"EXCEPTION in /process: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'prediction': 'Neutral', 'scores': {}})

@app.route('/stop', methods=['POST'])
def stop():
    global cap, stream, visual_window, audio_window, audio_data, latest_frame, prev_frame
    try:
        if cap is not None:
            if hasattr(cap, 'release'):
                cap.release()
            cap = None
        if stream is not None:
            try:
                stream.stop()
            except:
                pass
            stream = None
        visual_window = []
        audio_window = []
        audio_data = []
        latest_frame = None
        prev_frame = None
        return jsonify({'status': 'stopped'})
    except Exception as e:
        print(f"Error in stop: {e}")
        return jsonify({'status': 'stopped', 'error': str(e)})

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'message': 'MirrorMe Backend API is running',
        'models_loaded': {
            'no_glasses': svm_no is not None,
            'glasses': svm_glasses is not None,
            'glasses_detector': glasses_svm is not None
        }
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)