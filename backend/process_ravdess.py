import cv2
import librosa
import numpy as np
import os
from extract_features import extract_visual_features, extract_audio_features

def get_label(filename):
    try:
        parts = filename.split('-')
        if len(parts) < 3:
            print(f"  Warning: Unexpected filename format: {filename}, defaulting to Neutral")
            return 2
        emotion = int(parts[2])
        if emotion in [1, 2, 3]: 
            return 0  # Confident
        elif emotion in [4, 5, 6]: 
            return 1  # Nervous
        else: 
            return 2  # Neutral
    except (ValueError, IndexError) as e:
        print(f"  Warning: Error parsing label from {filename}: {e}, defaulting to Neutral")
        return 2

features_list = []
labels = []
glasses_labels = []
all_feature_keys = set()  # Collect all possible feature keys

video_dir = '../data/ravdess/video/Actor_01/Actor_01'  # Your path
video_dir = os.path.abspath(video_dir)  # Convert to absolute path
print(f"Video directory absolute path: {video_dir}")

if not os.path.exists(video_dir):
    print(f"Error: Video directory does not exist: {video_dir}")
    exit(1)

files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
print(f"Found {len(files)} video files to process")

if len(files) == 0:
    print("No .mp4 files found in directory")
    exit(1)

for idx, file in enumerate(files, 1):
    try:
        print(f"Processing file {idx}/{len(files)}: {file}")
        video_path = os.path.join(video_dir, file)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  Warning: Could not open {file}, skipping")
            continue
        
        # Extract audio
        try:
            y, sr = librosa.load(video_path, sr=16000)
        except Exception as e:
            print(f"  Warning: Audio extraction failed for {file}: {e}, using silence")
            y = np.zeros(16000)
            sr = 16000
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # Default FPS
        frame_duration = int(sr / fps)
        
        prev_frame = None
        visual_window = []
        audio_window = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                visual_feats, visual_window = extract_visual_features(frame, prev_frame, visual_window)
                start = frame_count * frame_duration
                end = min((frame_count + 1) * frame_duration, len(y))
                audio_chunk = y[start:end] if end > start else np.zeros(frame_duration)
                
                # Ensure audio_chunk is the right length
                if len(audio_chunk) == 0:
                    audio_chunk = np.zeros(frame_duration)
                elif len(audio_chunk) < frame_duration:
                    audio_chunk = np.pad(audio_chunk, (0, frame_duration - len(audio_chunk)), 'constant')
                
                audio_feats, audio_window = extract_audio_features(audio_chunk, sr, audio_window)
                combined = {**visual_feats, **audio_feats}
                
                # Collect all feature keys for consistency
                all_feature_keys.update(combined.keys())
                
                features_list.append(combined)
                labels.append(get_label(file))
                glasses_labels.append(1 if visual_feats.get('glasses', False) else 0)
                prev_frame = frame
                frame_count += 1
            except Exception as e:
                print(f"  Warning: Error processing frame {frame_count} of {file}: {e}")
                continue
        
        cap.release()
        print(f"  Finished processing {file}: {frame_count} frames")
    except Exception as e:
        print(f"  Error processing {file}: {e}, skipping")
        continue

if len(features_list) == 0:
    print("Error: No features extracted. Check video files and processing.")
    exit(1)

# Convert feature dictionaries to consistent vectors
sorted_keys = sorted(all_feature_keys)
print(f"Total unique features: {len(sorted_keys)}")
features_array = np.array([[f.get(k, 0) for k in sorted_keys] for f in features_list])
labels_array = np.array(labels)
glasses_array = np.array(glasses_labels)

output_path = '../data/ravdess_features.npz'
output_path = os.path.abspath(output_path)
np.savez(output_path, features=features_array, labels=labels_array, glasses=glasses_array, feature_names=sorted_keys)
print(f"\nProcessed {len(features_list)} frames from {len(set(labels))} emotion classes.")
print(f"Features shape: {features_array.shape}")
print(f"Saved to {output_path}")
print("Ready for training.")