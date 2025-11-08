# Dataset Links for MirrorMe

This document contains all the dataset links used in the MirrorMe project.

## Main Dataset

### RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

**Primary Dataset** - Used for training the main emotion recognition models.

- **Main Page**: https://zenodo.org/records/1188976
- **Audio Dataset** (Actors 01-24, ~215 MB):
  - Direct Download: https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1
- **Video Dataset** (Actor 01, ~553 MB):
  - Direct Download: https://zenodo.org/records/1188976/files/Video_Speech_Actor_01.zip?download=1
- **Full Video Dataset** (All Actors):
  - Available at: https://zenodo.org/records/1188976
  - Note: The project currently uses Actor_01 video data, but full dataset is available

**Citation:**
```
Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PLOS ONE, 13(5), e0196391.
```

## Optional Hand Gesture Datasets

### HGM-4 (Hand Gesture Multi-camera Dataset)

**Recommended for Quick Testing** - Smallest dataset, easiest to get started.

- **Link**: https://data.mendeley.com/datasets/jzy8zngkbg/8
- **Size**: ~50-100 MB
- **Format**: Images (JPG)
- **Gestures**: 26 (A-Z alphabet)
- **Note**: Requires free Mendeley account to download

### LeapGestRecog

**Balanced Size/Quality** - Good middle ground.

- **Link**: https://www.kaggle.com/datasets/gti-upm/leapgestrecog
- **Size**: ~1.2 GB
- **Format**: Images
- **Gestures**: 10
- **Note**: Requires Kaggle account

### HaGRID (Hand Gesture Recognition Image Dataset)

**Production Quality** - Best quality but very large.

- **Link**: https://github.com/hukenovs/hagrid
- **Size**: ~716 GB
- **Format**: Images
- **Gestures**: 18
- **Note**: Large download, best for production use

### IPN Hand

**Video Format** - For video-based gesture analysis.

- **Link**: https://gibranbenitez.github.io/IPN_Hand/
- **Size**: ~35 GB
- **Format**: Videos
- **Gestures**: 13
- **Note**: Video format requires different processing

## Additional Resources

### dlib Shape Predictor

The project uses dlib's 68-point facial landmark predictor:

- **File**: `shape_predictor_68_face_landmarks.dat`
- **Download**: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
- **Size**: ~99 MB (compressed), ~95 MB (uncompressed)

## Usage Notes

1. **RAVDESS is required** - This is the main dataset used for training
2. **Hand gesture datasets are optional** - The system works without them, but accuracy improves with them
3. **HGM-4 is recommended** for first-time users due to its small size
4. All datasets should be placed in the `data/` directory as specified in the README

## Download Scripts

The project includes helper scripts for downloading datasets:

- `backend/download_ravdess.py` - Downloads RAVDESS audio and video
- `backend/extract_bz2.py` - Extracts compressed dlib shape predictor

