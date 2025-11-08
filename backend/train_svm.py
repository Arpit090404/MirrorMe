from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import os

# Load processed data (try combined dataset first, fallback to RAVDESS only)
combined_path = os.path.abspath('../data/combined_features.npz')
ravdess_path = os.path.abspath('../data/ravdess_features.npz')

if os.path.exists(combined_path):
    data_path = combined_path
    print("Using combined dataset (RAVDESS + Hand Gestures)")
elif os.path.exists(ravdess_path):
    data_path = ravdess_path
    print("Using RAVDESS dataset only (add hand gestures for better accuracy)")
else:
    print(f"Error: No feature files found!")
    print(f"  Checked: {combined_path}")
    print(f"  Checked: {ravdess_path}")
    print("Please run process_ravdess.py first to extract features.")
    exit(1)

data = np.load(data_path, allow_pickle=True)
features = data['features']
labels = data['labels']
glasses = data['glasses']
feature_names = data.get('feature_names', None)

print(f"Loaded features: {features.shape}")
print(f"Labels: {labels.shape}, Classes: {len(np.unique(labels))}")
print(f"Glasses distribution: {np.bincount(glasses.astype(int))}")
if feature_names is not None:
    print(f"Feature names: {len(feature_names)} features")
else:
    print("Warning: No feature names found in saved data")

# Split for glasses/no-glasses
features_no = features[glasses == 0]
labels_no = labels[glasses == 0]
features_glasses = features[glasses == 1]
labels_glasses = labels[glasses == 1]

def train_svm(feats, lbls, glasses=False):
    if len(feats) == 0:
        print(f"No data available for {'glasses' if glasses else 'no-glasses'} model. Skipping.")
        return
    
    unique_classes = len(np.unique(lbls))
    if len(feats) < 2 or unique_classes < 2: 
        print(f"Insufficient data for {'glasses' if glasses else 'no-glasses'} (have {len(feats)} samples, {unique_classes} classes). Need at least 2 samples/classes. Skipping.")
        return
    
    # Remove NaN and Inf values
    mask = ~(np.isnan(feats).any(axis=1) | np.isinf(feats).any(axis=1))
    feats = feats[mask]
    lbls = lbls[mask]
    
    if len(feats) < 2:
        print(f"After cleaning, insufficient data for {'glasses' if glasses else 'no-glasses'}. Skipping.")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(feats, lbls, test_size=0.2, random_state=42, stratify=lbls if unique_classes > 1 else None)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    svm = GridSearchCV(SVC(), param_grid, cv=min(5, len(X_train)//2), error_score='raise')
    svm.fit(X_train, y_train)
    
    accuracy = svm.score(X_test, y_test)
    print(f"{'Glasses' if glasses else 'No-glasses'} model accuracy: {accuracy:.3f}")
    
    # Save models
    models_dir = '../models'
    models_dir = os.path.abspath(models_dir)
    os.makedirs(models_dir, exist_ok=True)
    
    model_name = 'svm_glasses.pkl' if glasses else 'svm_no_glasses.pkl'
    scaler_name = f'scaler_{"glasses" if glasses else "no"}.pkl'
    
    joblib.dump(svm, os.path.join(models_dir, model_name))
    joblib.dump(scaler, os.path.join(models_dir, scaler_name))
    print(f"  Saved model: {os.path.join(models_dir, model_name)}")
    print(f"  Saved scaler: {os.path.join(models_dir, scaler_name)}")
    return scaler, svm

train_svm(features_no, labels_no, glasses=False)
train_svm(features_glasses, labels_glasses, glasses=True)

# Save feature names globally for runtime inference
if feature_names is not None:
    models_dir = '../models'
    models_dir = os.path.abspath(models_dir)
    os.makedirs(models_dir, exist_ok=True)
    import json
    with open(os.path.join(models_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names.tolist() if isinstance(feature_names, np.ndarray) else feature_names, f)
    print(f"\nSaved feature names to {models_dir}/feature_names.json")

# Glasses detection SVM (placeholder - can be improved later)
models_dir = '../models'
models_dir = os.path.abspath(models_dir)
os.makedirs(models_dir, exist_ok=True)

if len(features) > 10:
    try:
        # Use a subset for glasses detection training
        sample_size = min(1000, len(features))
        indices = np.random.choice(len(features), sample_size, replace=False)
        sample_feats = features[indices]
        sample_glasses = glasses[indices]
        
        glasses_svm = SVC(kernel='linear')
        glasses_svm.fit(sample_feats, sample_glasses)
        joblib.dump(glasses_svm, os.path.join(models_dir, 'glasses_svm.pkl'))
        print(f"Glasses detection model saved to {os.path.join(models_dir, 'glasses_svm.pkl')}")
    except Exception as e:
        print(f"Warning: Could not train glasses detection model: {e}")

print(f"\nTraining complete. Models saved to {models_dir}/")