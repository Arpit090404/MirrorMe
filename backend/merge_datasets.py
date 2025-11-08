"""
Merge RAVDESS and hand gesture datasets for combined training.
"""

import numpy as np
import os

def merge_datasets(ravdess_file='ravdess_features.npz', 
                   hand_gesture_file='hand_gesture_features.npz',
                   output_file='combined_features.npz'):
    """
    Merge features from RAVDESS and hand gesture datasets.
    Ensures feature alignment between datasets.
    """
    print("Loading RAVDESS dataset...")
    ravdess_path = os.path.join('../data', ravdess_file)
    ravdess_path = os.path.abspath(ravdess_path)
    
    if not os.path.exists(ravdess_path):
        print(f"Error: {ravdess_path} not found!")
        return
    
    ravdess_data = np.load(ravdess_path, allow_pickle=True)
    ravdess_features = ravdess_data['features']
    ravdess_labels = ravdess_data['labels']
    ravdess_glasses = ravdess_data.get('glasses', np.zeros(len(ravdess_labels)))
    ravdess_keys = ravdess_data.get('feature_names', None)
    
    print(f"  RAVDESS: {ravdess_features.shape} features, {len(ravdess_labels)} samples")
    
    print("\nLoading hand gesture dataset...")
    hand_path = os.path.join('../data', hand_gesture_file)
    hand_path = os.path.abspath(hand_path)
    
    if not os.path.exists(hand_path):
        print(f"Warning: {hand_path} not found! Using only RAVDESS.")
        print("To add hand gestures, run: python process_hand_gestures.py <format> <data_dir>")
        # Use RAVDESS only
        output_path = os.path.join('../data', output_file)
        output_path = os.path.abspath(output_path)
        np.savez(output_path, 
                 features=ravdess_features,
                 labels=ravdess_labels,
                 glasses=ravdess_glasses,
                 feature_names=ravdess_keys)
        print(f"\nSaved RAVDESS-only dataset to {output_path}")
        return
    
    hand_data = np.load(hand_path, allow_pickle=True)
    hand_features = hand_data['features']
    hand_labels = hand_data['labels']
    hand_keys = hand_data.get('feature_names', None)
    
    print(f"  Hand Gestures: {hand_features.shape} features, {len(hand_labels)} samples")
    
    # Align feature keys - convert to lists if they're numpy arrays
    if ravdess_keys is None:
        ravdess_keys = [str(i) for i in range(ravdess_features.shape[1])]
    else:
        # Convert numpy array to list if needed
        if isinstance(ravdess_keys, np.ndarray):
            ravdess_keys = ravdess_keys.tolist()
        elif not isinstance(ravdess_keys, list):
            ravdess_keys = list(ravdess_keys)
        # Convert all to strings for consistency
        ravdess_keys = [str(k) for k in ravdess_keys]
    
    if hand_keys is None:
        hand_keys = [str(i) for i in range(hand_features.shape[1])]
    else:
        # Convert numpy array to list if needed
        if isinstance(hand_keys, np.ndarray):
            hand_keys = hand_keys.tolist()
        elif not isinstance(hand_keys, list):
            hand_keys = list(hand_keys)
        # Convert all to strings for consistency
        hand_keys = [str(k) for k in hand_keys]
    
    # Get union of all feature keys (now both are lists of strings)
    all_keys = sorted(set(ravdess_keys) | set(hand_keys))
    print(f"\nTotal unique features: {len(all_keys)}")
    print(f"  RAVDESS features: {len(ravdess_keys)}")
    print(f"  Hand gesture features: {len(hand_keys)}")
    print(f"  Overlapping features: {len(set(ravdess_keys) & set(hand_keys))}")
    
    # Align RAVDESS features
    ravdess_aligned = np.zeros((len(ravdess_labels), len(all_keys)))
    for i, key in enumerate(all_keys):
        if key in ravdess_keys:
            idx = ravdess_keys.index(key)  # Both are lists now
            ravdess_aligned[:, i] = ravdess_features[:, idx]
    
    # Align hand gesture features
    hand_aligned = np.zeros((len(hand_labels), len(all_keys)))
    for i, key in enumerate(all_keys):
        if key in hand_keys:
            idx = hand_keys.index(key)  # Both are lists now
            hand_aligned[:, i] = hand_features[:, idx]
    
    # Combine datasets
    combined_features = np.vstack([ravdess_aligned, hand_aligned])
    combined_labels = np.hstack([ravdess_labels, hand_labels])
    combined_glasses = np.hstack([ravdess_glasses, np.zeros(len(hand_labels))])  # Assume no glasses info for hand dataset
    
    print(f"\nCombined dataset:")
    print(f"  Total samples: {len(combined_labels)}")
    print(f"  Feature dimension: {combined_features.shape[1]}")
    print(f"  Classes: {len(np.unique(combined_labels))}")
    
    # Save combined dataset
    output_path = os.path.join('../data', output_file)
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.savez(output_path,
             features=combined_features,
             labels=combined_labels,
             glasses=combined_glasses,
             feature_names=all_keys)
    
    print(f"\nSaved combined dataset to {output_path}")
    print("Ready for training with: python train_svm.py")

if __name__ == '__main__':
    import sys
    
    hand_file = sys.argv[1] if len(sys.argv) > 1 else 'hand_gesture_features.npz'
    merge_datasets(hand_gesture_file=hand_file)

