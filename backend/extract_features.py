import cv2
import dlib
import numpy as np
import librosa
from scipy.spatial import distance as dist

import os

# Load dlib face detector and landmark predictor
face_detector = dlib.get_frontal_face_detector()

# Try to find shape_predictor in current directory or backend directory
shape_predictor_path = os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat')
if not os.path.exists(shape_predictor_path):
    # Try current directory
    current_dir_path = 'shape_predictor_68_face_landmarks.dat'
    if os.path.exists(current_dir_path):
        shape_predictor_path = current_dir_path
    else:
        raise FileNotFoundError(
            f"shape_predictor_68_face_landmarks.dat not found in {os.path.dirname(__file__)}. "
            f"Please download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 "
            f"and place it in the backend directory."
        )

landmark_predictor = dlib.shape_predictor(shape_predictor_path)
haar_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Otsu Thresholding for binary mask
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Canny Edge Detection
    edges = cv2.Canny(gray, 100, 200)
    # CLAHE for lighting robustness
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return enhanced, edges, thresh

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

def get_head_pose(shape):
    try:
        image_points = np.array([shape[30], shape[8], shape[36], shape[45], shape[48], shape[54]], dtype="double")
        model_points = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
                                 (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)])
        camera_matrix = np.eye(3)
        dist_coeffs = np.zeros((4,1))
        # RANSAC for robust pose estimation (returns 4 values: retval, rvec, tvec, inliers)
        result = cv2.solvePnPRansac(model_points, image_points, camera_matrix, dist_coeffs)
        if len(result) == 4:
            retval, rotation_vector, tvec, inliers = result
            if retval and rotation_vector is not None:
                return rotation_vector
        elif len(result) == 3:
            retval, rotation_vector, tvec = result
            if retval and rotation_vector is not None:
                return rotation_vector
    except Exception:
        pass
    return np.zeros((3,1))  # Fallback

def detect_hand_motion(frame, prev_frame, edges, thresh):
    """Enhanced hand gesture detection with multiple feature extraction"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Improved skin color range for better detection
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    # Combine with Otsu thresh for better mask
    mask = cv2.bitwise_and(mask, thresh)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    # Enhanced hand detection features
    hand_edges = cv2.Canny(mask, 50, 150)
    contours, _ = cv2.findContours(hand_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate hand area from valid contours
    valid_contours = [c for c in contours if cv2.contourArea(c) > 1000]
    hand_area = sum(cv2.contourArea(c) for c in valid_contours)
    
    # Hand gesture features
    hand_features = {
        'hand_area': hand_area,
        'num_contours': len(valid_contours),
        'convexity_defects': 0,
        'hand_circularity': 0,
        'finger_count': 0
    }
    
    # Analyze largest contour for gesture features
    if len(valid_contours) > 0:
        largest_contour = max(valid_contours, key=cv2.contourArea)
        
        # Convexity defects (indicates finger positions)
        hull = cv2.convexHull(largest_contour, returnPoints=False)
        if len(hull) > 3:
            defects = cv2.convexityDefects(largest_contour, hull)
            if defects is not None:
                hand_features['convexity_defects'] = len(defects)
                # Estimate finger count (rough)
                hand_features['finger_count'] = min(5, len(defects))
        
        # Circularity (compactness measure)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter > 0:
            hand_features['hand_circularity'] = 4 * np.pi * area / (perimeter ** 2)
    
    # Motion energy calculation
    motion_energy = 0
    hand_motion = 0
    if prev_frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Global motion energy (Farneback dense optical flow)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_energy = np.mean(mag)
        
        # Hand-specific motion (only in hand regions)
        if len(valid_contours) > 0:
            hand_mask = np.zeros_like(mask)
            cv2.drawContours(hand_mask, valid_contours, -1, 255, -1)
            hand_flow = flow[hand_mask > 0]
            if len(hand_flow) > 0:
                hand_motion = np.mean(np.linalg.norm(hand_flow, axis=1))
    
    # Blob detection for hand regions
    hand_blobs = 0
    if len(valid_contours) > 0:
        # Simple blob detection: count distinct hand regions
        hand_blobs = len(valid_contours)
        # Advanced: Use cv2.SimpleBlobDetector if needed
    
    # Image moments for hand shape analysis
    hand_moments_m00 = 0.0
    hand_moments_m01 = 0.0
    hand_moments_m10 = 0.0
    if len(valid_contours) > 0:
        largest_contour = max(valid_contours, key=cv2.contourArea)
        moments = cv2.moments(largest_contour)
        hand_moments_m00 = moments['m00'] if moments['m00'] != 0 else 0.0
        hand_moments_m01 = moments['m01'] if moments['m01'] != 0 else 0.0
        hand_moments_m10 = moments['m10'] if moments['m10'] != 0 else 0.0
    
    # Color histogram analysis for hand detection quality
    color_hist_std = 0.0
    if len(valid_contours) > 0:
        hand_mask = np.zeros_like(mask)
        cv2.drawContours(hand_mask, valid_contours, -1, 255, -1)
        hand_region = cv2.bitwise_and(frame, frame, mask=hand_mask)
        if hand_region.size > 0:
            # Calculate color histogram standard deviation (measures color diversity)
            hist_b = cv2.calcHist([hand_region], [0], hand_mask, [256], [0, 256])
            hist_g = cv2.calcHist([hand_region], [1], hand_mask, [256], [0, 256])
            hist_r = cv2.calcHist([hand_region], [2], hand_mask, [256], [0, 256])
            color_hist_std = float(np.std(hist_b) + np.std(hist_g) + np.std(hist_r)) / 3.0
    
    hand_features['motion_energy'] = motion_energy
    hand_features['hand_motion'] = hand_motion
    hand_features['hand_blobs'] = hand_blobs
    hand_features['hand_moments_m00'] = hand_moments_m00
    hand_features['hand_moments_m01'] = hand_moments_m01
    hand_features['hand_moments_m10'] = hand_moments_m10
    hand_features['color_histogram_std'] = color_hist_std
    
    return hand_features

def extract_lbp(image):
    if len(image.shape) == 3 and image.shape[2] == 1:
        gray = image[:,:,0]  # Handle single-channel color image
    elif len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        gray = image
    else:
        return np.zeros(256)  # Fallback for invalid input
    lbp = np.zeros_like(gray, dtype=np.uint8)
    for i in range(1, gray.shape[0]-1):
        for j in range(1, gray.shape[1]-1):
            center = gray[i, j]
            code = 0
            code |= (gray[i-1, j-1] >= center) << 7
            code |= (gray[i-1, j] >= center) << 6
            code |= (gray[i-1, j+1] >= center) << 5
            code |= (gray[i, j+1] >= center) << 4
            code |= (gray[i+1, j+1] >= center) << 3
            code |= (gray[i+1, j] >= center) << 2
            code |= (gray[i+1, j-1] >= center) << 1
            code |= (gray[i, j-1] >= center) << 0
            lbp[i, j] = code
    hist = cv2.calcHist([lbp], [0], None, [256], [0, 256]).flatten()
    return hist / (hist.sum() + 1e-7)

def detect_glasses(frame, shape, enhanced):
    if shape is None:
        return False
    # Ensure eye_region slice is valid
    y1, y2, x1, x2 = shape[36][1]-10, shape[45][1]+10, shape[36][0]-10, shape[45][0]+10
    y1, y2 = max(0, y1), min(enhanced.shape[0], y2)
    x1, x2 = max(0, x1), min(enhanced.shape[1], x2)
    eye_region = enhanced[y1:y2, x1:x2]
    if eye_region.size == 0 or eye_region.shape[0] == 0 or eye_region.shape[1] == 0:
        return False
    lbp_features = extract_lbp(eye_region).reshape(1, -1)
    variance = np.var(eye_region)
    return variance > 50  # Fallback; replace with SVM

def extract_visual_features(frame, prev_frame=None, visual_window=[]):
    enhanced, edges, thresh = preprocess_frame(frame)
    faces = face_detector(enhanced, 1)
    if len(faces) == 0:
        haar_faces = haar_face_cascade.detectMultiScale(enhanced, 1.3, 5)
        if len(haar_faces) > 0:
            # Convert Haar cascade format to dlib rectangle format
            x, y, w, h = haar_faces[0]
            faces = [dlib.rectangle(x, y, x+w, y+h)]
    
    # Initialize default features (including enhanced hand gesture features)
    features = {
        'harris_corners': 0,
        'shi_tomasi_corners': 0,
        'sift_keypoints': 0,
        'orb_keypoints': 0,
        'blink': 0,
        'smile': 0,
        'head_yaw': 0.0,
        'head_pitch': 0.0,
        'head_roll': 0.0,
        'gaze_offset': 0.0,
        'iris_circles': 0,
        'hand_area': 0.0,
        'motion_energy': 0.0,
        'hand_motion': 0.0,
        'num_contours': 0,
        'convexity_defects': 0,
        'hand_circularity': 0.0,
        'finger_count': 0,
        'hand_blobs': 0,
        'hand_moments_m00': 0.0,
        'hand_moments_m01': 0.0,
        'hand_moments_m10': 0.0,
        'color_histogram_std': 0.0,
        'glasses': False
    }
    
    shape = None
    if len(faces) > 0:
        try:
            face = faces[0]
            shape = landmark_predictor(enhanced, face)
            shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
            
            # Harris corners for gaze/smile enhancement
            harris = cv2.cornerHarris(enhanced, 2, 3, 0.04)
            features['harris_corners'] = np.sum(harris > 0.01 * harris.max())
            
            # Shi-Tomasi corner detection (better than Harris for tracking)
            corners = cv2.goodFeaturesToTrack(enhanced, maxCorners=100, qualityLevel=0.01, minDistance=10)
            features['shi_tomasi_corners'] = len(corners) if corners is not None else 0
            
            # SIFT keypoints for gesture/head
            sift = cv2.SIFT_create()
            kp, des = sift.detectAndCompute(enhanced, None)
            features['sift_keypoints'] = len(kp) if kp is not None else 0
            
            # ORB keypoints (faster alternative to SIFT)
            orb = cv2.ORB_create(nfeatures=100)
            orb_kp, orb_des = orb.detectAndCompute(enhanced, None)
            features['orb_keypoints'] = len(orb_kp) if orb_kp is not None else 0
            
            left_eye = shape[36:42]
            right_eye = shape[42:48]
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            features['blink'] = 1 if ear < 0.2 else 0
            
            mouth = shape[48:68]
            mar = mouth_aspect_ratio(mouth)
            features['smile'] = 1 if mar > 0.4 else 0
            
            rotation = get_head_pose(shape)
            if rotation is not None and len(rotation) >= 3:
                features['head_yaw'] = float(rotation[1][0]) if rotation.shape[0] > 1 else 0.0
                features['head_pitch'] = float(rotation[0][0]) if rotation.shape[0] > 0 else 0.0
                features['head_roll'] = float(rotation[2][0]) if rotation.shape[0] > 2 else 0.0
            
            left_iris = np.mean(shape[36:42], axis=0)
            features['gaze_offset'] = float(left_iris[0] - shape[30][0])
            
            # Hough for iris circles in gaze - with bounds checking
            try:
                y1, y2 = int(shape[36][1]), int(shape[45][1])
                x1, x2 = int(shape[36][0]), int(shape[45][0])
                y1, y2 = max(0, min(y1, y2)), min(enhanced.shape[0], max(y1, y2))
                x1, x2 = max(0, min(x1, x2)), min(enhanced.shape[1], max(x1, x2))
                
                if y2 > y1 and x2 > x1:
                    eye_region_gray = enhanced[y1:y2, x1:x2]
                    if eye_region_gray.size > 0:
                        circles = cv2.HoughCircles(eye_region_gray, cv2.HOUGH_GRADIENT, 1.2, 20, param1=50, param2=30, minRadius=5, maxRadius=15)
                        features['iris_circles'] = len(circles[0]) if circles is not None and len(circles) > 0 else 0
            except Exception:
                features['iris_circles'] = 0
            
            # Enhanced hand gesture features
            hand_features = detect_hand_motion(frame, prev_frame, edges, thresh)
            features['hand_area'] = hand_features['hand_area']
            features['motion_energy'] = hand_features['motion_energy']
            features['hand_motion'] = hand_features['hand_motion']
            features['num_contours'] = hand_features['num_contours']
            features['convexity_defects'] = hand_features['convexity_defects']
            features['hand_circularity'] = hand_features['hand_circularity']
            features['finger_count'] = hand_features['finger_count']
            features['hand_blobs'] = hand_features['hand_blobs']
            features['hand_moments_m00'] = hand_features['hand_moments_m00']
            features['hand_moments_m01'] = hand_features['hand_moments_m01']
            features['hand_moments_m10'] = hand_features['hand_moments_m10']
            features['color_histogram_std'] = hand_features['color_histogram_std']
        except Exception as e:
            print(f"Error processing face landmarks: {e}")
    
    features['glasses'] = detect_glasses(frame, shape, enhanced)
    
    # Always extract hand features (even if no face detected)
    if 'hand_area' not in features or features.get('hand_area', 0) == 0:
        hand_features = detect_hand_motion(frame, prev_frame, edges, thresh)
        features['hand_area'] = hand_features['hand_area']
        features['motion_energy'] = hand_features['motion_energy']
        features['hand_motion'] = hand_features['hand_motion']
        features['num_contours'] = hand_features['num_contours']
        features['convexity_defects'] = hand_features['convexity_defects']
        features['hand_circularity'] = hand_features['hand_circularity']
        features['finger_count'] = hand_features['finger_count']
        features['hand_blobs'] = hand_features['hand_blobs']
        features['hand_moments_m00'] = hand_features['hand_moments_m00']
        features['hand_moments_m01'] = hand_features['hand_moments_m01']
        features['hand_moments_m10'] = hand_features['hand_moments_m10']
        features['color_histogram_std'] = hand_features['color_histogram_std']
    
    visual_window.append(features.copy())
    if len(visual_window) > 5:
        visual_window.pop(0)
    
    # Calculate temporal features (including hand gesture features and new CV features)
    temporal_keys = ['head_yaw', 'motion_energy', 'blink', 'sift_keypoints', 'harris_corners', 
                     'orb_keypoints', 'shi_tomasi_corners', 'hand_motion', 'hand_area', 
                     'finger_count', 'convexity_defects', 'hand_blobs']
    for key in temporal_keys:
        vals = [f.get(key, 0) for f in visual_window if key in f]
        if len(vals) > 0:
            features[f'{key}_mean'] = float(np.mean(vals))
            features[f'{key}_std'] = float(np.std(vals))
            features[f'{key}_slope'] = float(np.polyfit(range(len(vals)), vals, 1)[0]) if len(vals) > 1 else 0.0
        else:
            features[f'{key}_mean'] = 0.0
            features[f'{key}_std'] = 0.0
            features[f'{key}_slope'] = 0.0
    
    return features, visual_window

def extract_audio_features(audio_chunk, sr=16000, audio_window=[]):
    rms = librosa.feature.rms(y=audio_chunk)
    features = {'volume_mean': np.mean(rms), 'volume_std': np.std(rms)}
    pitch = librosa.yin(audio_chunk, fmin=75, fmax=500, sr=sr)
    features['pitch_mean'] = np.mean(pitch[pitch > 0]) if np.any(pitch > 0) else 0
    features['pitch_std'] = np.std(pitch[pitch > 0]) if np.any(pitch > 0) else 0
    hesitation = np.sum(rms < 0.01) / len(rms) if len(rms) > 0 else 0
    features['hesitation'] = hesitation
    
    audio_window.append(features.copy())
    if len(audio_window) > 5:
        audio_window.pop(0)
    
    # Calculate temporal features for audio
    keys = list(features.keys())
    for key in keys:
        if key.endswith('_slope'):  # Skip already computed slopes
            continue
        vals = [f.get(key, 0) for f in audio_window if key in f]
        if len(vals) > 1:
            features[f'{key}_slope'] = float(np.polyfit(range(len(vals)), vals, 1)[0])
        else:
            features[f'{key}_slope'] = 0.0
    
    return features, audio_window