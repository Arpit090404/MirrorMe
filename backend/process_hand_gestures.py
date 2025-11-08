import cv2
import numpy as np
import os
from extract_features import extract_visual_features, extract_audio_features

def process_hagrid_dataset(data_dir, output_file='hand_gesture_features.npz'):
    """
    Process HaGRID dataset format:
    data_dir/
      train/
        gesture_class/
          image1.jpg
          image2.jpg
          ...
    """
    features_list = []
    labels = []
    all_feature_keys = set()
    
    gesture_classes = {
        'call': 0,  # Confident gesture
        'dislike': 1,  # Nervous/negative
        'fist': 0,  # Could be confident
        'four': 0,  # Open/confident
        'like': 0,  # Positive/confident
        'mute': 1,  # Restrictive/nervous
        'ok': 0,  # Positive
        'one': 0,  # Neutral
        'peace': 0,  # Open/confident
        'rock': 0,  # Confident
        'stop': 1,  # Restrictive
        'stop_inverted': 1,
        'three': 0,  # Open
        'three2': 0,
        'two_up': 0,  # Open
        'two_up_inverted': 0,
        'palm': 0,  # Open/confident
        'no_gesture': 2  # Neutral/no gesture
    }
    
    data_path = os.path.abspath(data_dir)
    print(f"Processing HaGRID dataset from: {data_path}")
    
    train_dir = os.path.join(data_path, 'train')
    if not os.path.exists(train_dir):
        print(f"Warning: train directory not found, looking for gesture folders in root...")
        train_dir = data_path
    
    total_files = 0
    processed_files = 0
    
    for gesture_name, emotion_label in gesture_classes.items():
        gesture_dir = os.path.join(train_dir, gesture_name)
        if not os.path.exists(gesture_dir):
            print(f"  Skipping {gesture_name} - directory not found")
            continue
        
        image_files = [f for f in os.listdir(gesture_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_files += len(image_files)
        
        print(f"Processing {gesture_name}: {len(image_files)} images")
        
        for idx, img_file in enumerate(image_files[:1000]):  # Limit to 1000 per class for speed
            try:
                img_path = os.path.join(gesture_dir, img_file)
                frame = cv2.imread(img_path)
                if frame is None:
                    continue
                
                # Extract visual features
                visual_features, _ = extract_visual_features(frame, None, [])
                
                # Create dummy audio (zeros) since these are images
                audio_chunk = np.zeros(16000)
                audio_features, _ = extract_audio_features(audio_chunk, sr=16000, audio_window=[])
                
                combined = {**visual_features, **audio_features}
                all_feature_keys.update(combined.keys())
                
                features_list.append(combined)
                labels.append(emotion_label)
                processed_files += 1
                
                if (idx + 1) % 100 == 0:
                    print(f"    Processed {idx + 1}/{min(1000, len(image_files))} images")
            except Exception as e:
                print(f"  Error processing {img_file}: {e}")
                continue
    
    if len(features_list) == 0:
        print("Error: No features extracted. Check dataset path and structure.")
        return
    
    # Convert to consistent feature vectors
    sorted_keys = sorted(all_feature_keys)
    print(f"Total unique features: {len(sorted_keys)}")
    features_array = np.array([[f.get(k, 0) for k in sorted_keys] for f in features_list])
    labels_array = np.array(labels)
    
    output_path = os.path.join('../data', output_file)
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.savez(output_path, features=features_array, labels=labels_array, feature_names=sorted_keys)
    print(f"\nProcessed {processed_files} frames from {len(set(labels))} classes.")
    print(f"Features shape: {features_array.shape}")
    print(f"Saved to {output_path}")
    print("Ready to merge with RAVDESS features for training.")

def process_hgm4_dataset(data_dir, output_file='hand_gesture_features.npz', camera_view='Front_CAM'):
    """
    Process HGM-4 dataset format:
    data_dir/HGM-1.0/
      Front_CAM/  (or Below_CAM, Left_CAM, Right_CAM)
        A/
          P1_001.jpg, P1_002.jpg, ...
        B/
          ...
        Z/
          ...
    
    Args:
        data_dir: Path to HGM-1.0 folder (or parent folder containing HGM-1.0)
        output_file: Output filename
        camera_view: Which camera view to use ('Front_CAM', 'Below_CAM', 'Left_CAM', 'Right_CAM')
    """
    features_list = []
    labels = []
    all_feature_keys = set()
    
    # Map letters A-Z to emotion labels
    # Most gestures are neutral/confident, but some closed gestures might indicate nervousness
    # A-Z are mostly open/confident gestures (like pointing, showing numbers)
    gesture_mapping = {
        letter: 0 if letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' else 2
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    }
    # Some specific mappings for closed/restrictive gestures could be nervous
    gesture_mapping.update({
        'F': 0,  # Fist - could be confident
        'C': 0,  # C shape - open
    })
    
    data_path = os.path.abspath(data_dir)
    
    # Check if HGM-1.0 folder exists
    hgm_folder = os.path.join(data_path, 'HGM-1.0')
    if not os.path.exists(hgm_folder):
        # Maybe data_dir already points to HGM-1.0
        if os.path.basename(data_path) == 'HGM-1.0':
            hgm_folder = data_path
        else:
            print(f"Error: HGM-1.0 folder not found in {data_path}")
            return
    
    camera_path = os.path.join(hgm_folder, camera_view)
    if not os.path.exists(camera_path):
        print(f"Error: Camera view '{camera_view}' not found in {hgm_folder}")
        print(f"Available views: {[d for d in os.listdir(hgm_folder) if os.path.isdir(os.path.join(hgm_folder, d))]}")
        return
    
    print(f"Processing HGM-4 dataset from: {camera_path}")
    print(f"Using camera view: {camera_view}")
    
    total_files = 0
    processed_files = 0
    
    # Process each gesture folder (A-Z)
    gesture_folders = sorted([d for d in os.listdir(camera_path) 
                              if os.path.isdir(os.path.join(camera_path, d)) and len(d) == 1])
    
    print(f"Found {len(gesture_folders)} gesture folders: {gesture_folders}")
    
    for gesture_letter in gesture_folders:
        gesture_dir = os.path.join(camera_path, gesture_letter)
        emotion_label = gesture_mapping.get(gesture_letter, 2)  # Default to Neutral
        
        image_files = sorted([f for f in os.listdir(gesture_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        total_files += len(image_files)
        
        print(f"Processing gesture '{gesture_letter}': {len(image_files)} images (label: {emotion_label})")
        
        # Process all images (or limit for speed)
        for idx, img_file in enumerate(image_files):
            try:
                img_path = os.path.join(gesture_dir, img_file)
                frame = cv2.imread(img_path)
                if frame is None:
                    continue
                
                # Extract visual features
                visual_features, _ = extract_visual_features(frame, None, [])
                
                # Create dummy audio (zeros) since these are images
                audio_chunk = np.zeros(16000)
                audio_features, _ = extract_audio_features(audio_chunk, sr=16000, audio_window=[])
                
                combined = {**visual_features, **audio_features}
                all_feature_keys.update(combined.keys())
                
                features_list.append(combined)
                labels.append(emotion_label)
                processed_files += 1
                
                if (idx + 1) % 100 == 0:
                    print(f"    Processed {idx + 1}/{len(image_files)} images")
            except Exception as e:
                print(f"  Error processing {img_file}: {e}")
                continue
    
    if len(features_list) == 0:
        print("Error: No features extracted. Check dataset path and structure.")
        return
    
    # Convert to consistent feature vectors
    sorted_keys = sorted(all_feature_keys)
    print(f"Total unique features: {len(sorted_keys)}")
    features_array = np.array([[f.get(k, 0) for k in sorted_keys] for f in features_list])
    labels_array = np.array(labels)
    
    output_path = os.path.join('../data', output_file)
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.savez(output_path, features=features_array, labels=labels_array, feature_names=sorted_keys)
    print(f"\nProcessed {processed_files} frames from {len(set(labels))} classes.")
    print(f"Features shape: {features_array.shape}")
    print(f"Saved to {output_path}")
    print("Ready to merge with RAVDESS features for training.")

def process_video_dataset(data_dir, output_file='hand_gesture_video_features.npz'):
    """
    Process video-based hand gesture datasets (like IPN Hand Dataset).
    Format: data_dir/gesture_class/video1.mp4, video2.mp4, ...
    """
    features_list = []
    labels = []
    all_feature_keys = set()
    
    # Map gesture names to emotion labels (adjust based on your dataset)
    gesture_classes = {}  # Will be populated from directory structure
    
    data_path = os.path.abspath(data_dir)
    print(f"Processing video dataset from: {data_path}")
    
    video_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_files.append(os.path.join(root, file))
    
    print(f"Found {len(video_files)} video files")
    
    for idx, video_path in enumerate(video_files[:200]):  # Limit for initial processing
        try:
            print(f"Processing {idx + 1}/{min(200, len(video_files))}: {os.path.basename(video_path)}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue
            
            # Extract audio if possible
            try:
                import librosa
                y, sr = librosa.load(video_path, sr=16000)
            except:
                y = np.zeros(16000)
                sr = 16000
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_duration = int(sr / fps)
            
            prev_frame = None
            visual_window = []
            audio_window = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 10th frame for efficiency
                if frame_count % 10 == 0:
                    visual_features, visual_window = extract_visual_features(frame, prev_frame, visual_window)
                    
                    start = frame_count * frame_duration
                    end = min((frame_count + 1) * frame_duration, len(y))
                    audio_chunk = y[start:end] if end > start else np.zeros(frame_duration)
                    if len(audio_chunk) < frame_duration:
                        audio_chunk = np.pad(audio_chunk, (0, frame_duration - len(audio_chunk)))
                    
                    audio_features, audio_window = extract_audio_features(audio_chunk, sr=sr, audio_window=audio_window)
                    
                    combined = {**visual_features, **audio_features}
                    all_feature_keys.update(combined.keys())
                    
                    features_list.append(combined)
                    # Determine label from path or filename (adjust based on your dataset)
                    labels.append(2)  # Default to Neutral, update based on dataset structure
                
                prev_frame = frame
                frame_count += 1
            
            cap.release()
        except Exception as e:
            print(f"  Error processing {video_path}: {e}")
            continue
    
    if len(features_list) == 0:
        print("Error: No features extracted.")
        return
    
    sorted_keys = sorted(all_feature_keys)
    features_array = np.array([[f.get(k, 0) for k in sorted_keys] for f in features_list])
    labels_array = np.array(labels)
    
    output_path = os.path.join('../data', output_file)
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.savez(output_path, features=features_array, labels=labels_array, feature_names=sorted_keys)
    print(f"\nProcessed {len(features_list)} frames.")
    print(f"Features shape: {features_array.shape}")
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    import sys
    
    print("=" * 60)
    print("Hand Gesture Dataset Processor")
    print("=" * 60)
    print("\nSupported Formats:")
    print("1. HGM-4: Multi-camera hand gesture dataset (A-Z gestures)")
    print("   - Run: python process_hand_gestures.py hgm4 data/small_gestures/HGM-1.0")
    print("\n2. HaGRID: https://github.com/hukenovs/hagrid")
    print("   - Run: python process_hand_gestures.py hagrid data/hagrid")
    print("\n3. Video-based: IPN Hand, etc.")
    print("   - Run: python process_hand_gestures.py video data/ipn_hand")
    print("=" * 60)
    
    if len(sys.argv) < 3:
        print("\nUsage:")
        print("  python process_hand_gestures.py <format> <data_dir> [camera_view]")
        print("\nFormats:")
        print("  hgm4   - For HGM-4 dataset format")
        print("  hagrid - For HaGRID dataset format")
        print("  video  - For video-based datasets")
        print("\nFor HGM-4, optional camera_view: Front_CAM (default), Below_CAM, Left_CAM, Right_CAM")
        sys.exit(1)
    
    format_type = sys.argv[1].lower()
    data_dir = sys.argv[2]
    camera_view = sys.argv[3] if len(sys.argv) > 3 else 'Front_CAM'
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist!")
        sys.exit(1)
    
    if format_type == 'hgm4':
        process_hgm4_dataset(data_dir, camera_view=camera_view)
    elif format_type == 'hagrid':
        process_hagrid_dataset(data_dir)
    elif format_type == 'video':
        process_video_dataset(data_dir)
    else:
        print(f"Error: Unknown format '{format_type}'. Use 'hgm4', 'hagrid', or 'video'")
        sys.exit(1)

