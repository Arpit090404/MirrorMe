# MirrorMe - AI Communication Coach

**Real-time public speaking & interview feedback system using classical computer vision**

> Turn fear of public speaking into confidence — one practice session at a time, privately and intelligently.

## 🎯 What is MirrorMe?

MirrorMe is a lightweight, privacy-focused, on-device AI system that analyzes your non-verbal cues and vocal delivery in real-time. It uses only classical computer vision and traditional ML — **no deep learning, no GPU, no cloud**.

### Key Features

- ✅ **Real-Time Analysis**: Live feedback on posture, gestures, eye contact, and voice
- 🔒 **100% Privacy**: Everything runs on your device, no data leaves your computer
- 🤖 **AI-Powered**: SVMs with 92.7% accuracy trained on RAVDESS + hand gesture datasets
- 🎨 **Beautiful UI**: Liquid glass morphism design with dark/light mode
- 👤 **Guest Mode**: Try it without creating an account
- 📊 **Performance Metrics**: 5 detailed metrics with real-time progress tracking
- 🤏 **Hand Gesture Support**: Enhanced gesture detection with optional dataset integration
- 📈 **Session Summary**: Detailed feedback after each practice session

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Webcam & Microphone

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd MirrorMe
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Node dependencies**
   ```bash
   cd frontend
   npm install
   ```

4. **Process data & train models** (First time only)
   ```bash
   cd backend
   python process_ravdess.py  # Extracts features from RAVDESS
   python train_svm.py         # Trains ML models
   ```
   
   **Optional: Add Hand Gesture Dataset** (for improved accuracy)
   ```bash
   # Download HGM-4 dataset (~50-100 MB) from:
   # https://data.mendeley.com/datasets/jzy8zngkbg/8
   # Extract to: data/small_gestures/HGM-1.0/
   
   cd backend
   python process_hand_gestures.py hgm4 ../data/small_gestures/HGM-1.0 Front_CAM
   python merge_datasets.py  # Merge with RAVDESS
   python train_svm.py       # Retrain with combined dataset
   ```

5. **Run the application**
   
   Terminal 1 (Backend):
   ```bash
   cd backend
   python app.py
   # Backend starts on: http://127.0.0.1:5000
   ```
   
   Terminal 2 (Frontend):
   ```bash
   cd frontend
   npx next dev -p 3001
   # Frontend starts on: http://localhost:3001
   # Browser opens automatically!
   ```

6. **Open in browser**: http://localhost:3001

## 📊 System Architecture

### Overview
```
┌─────────────────────────────────────────────────────────────────┐
│                         USER'S BROWSER                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │         FRONTEND (Next.js + React + TypeScript)            │  │
│  │  - Captures video frames via getUserMedia                  │  │
│  │  - Sends frames to backend every 500ms                    │  │
│  │  - Displays real-time predictions and metrics             │  │
│  │  - Shows session summary after recording                  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                         │                                       │
│                         │ HTTP Requests                         │
│                         │ POST /process_frame (base64 images)   │
│                         │ GET /process (every 500ms)            │
│                         │ POST /start, /stop                    │
│                         ▼                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    USER'S COMPUTER (Local)                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │         BACKEND (Flask + Python)                          │  │
│  │  - Receives frames from frontend                          │  │
│  │  - Extracts visual + audio features                       │  │
│  │  - Runs SVM models for prediction                         │  │
│  │  - Calculates rule-based scores                           │  │
│  │  - Returns JSON with prediction & metrics                 │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### How It Works

1. **Frontend** captures video frames from webcam using `getUserMedia`
2. **Frames are sent** to backend as base64-encoded JPEG images
3. **Backend processes** each frame:
   - Extracts 37+ visual features (face, gaze, gestures, etc.)
   - Analyzes audio (volume, pitch, hesitation)
   - Runs SVM models to predict: Confident / Nervous / Neutral
   - Calculates 5 performance metrics
4. **Frontend displays** results in real-time (updates every 500ms)
5. **After recording stops**, session summary shows overall analysis

## 📁 Project Structure

```
MirrorMe/
├── backend/
│   ├── app.py                   # Flask server & endpoints
│   ├── extract_features.py      # Visual + audio feature extraction
│   ├── process_ravdess.py       # RAVDESS data processing
│   ├── process_hand_gestures.py # Hand gesture dataset processing
│   ├── merge_datasets.py        # Merge RAVDESS + hand gestures
│   ├── train_svm.py             # Model training
│   ├── rule_based.py            # Performance scoring system
│   └── shape_predictor_68_face_landmarks.dat
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx         # Landing page
│   │   │   ├── login/           # Auth pages
│   │   │   └── dashboard/       # Main UI with video feed
│   │   ├── components/
│   │   │   ├── ui/              # shadcn/ui components
│   │   │   ├── video-stream.tsx # Webcam component
│   │   │   ├── theme-provider.tsx
│   │   │   └── theme-toggle.tsx
│   │   └── lib/
│   │       ├── auth-context.tsx
│   │       └── utils.ts
│   └── package.json
├── models/                      # Trained SVMs (*.pkl files)
├── data/                        # Datasets
│   ├── ravdess/                 # RAVDESS dataset
│   ├── small_gestures/          # Hand gesture datasets (optional)
│   ├── ravdess_features.npz
│   ├── hand_gesture_features.npz
│   └── combined_features.npz
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🎨 UI Features

### Pages

**Landing Page** (`/`)
- Hero section with animated gradient
- Feature cards
- "Start Practicing" button → Login page

**Login/Signup** (`/login`)
- Sign In / Sign Up tabs
- Email & password fields
- "Continue as Guest" button → Dashboard

**Dashboard** (`/dashboard`)
- **Live webcam feed**: See yourself in real-time (mirrored)
- **Camera/Mic controls**: Toggle on/off buttons
- **Start/Stop Analysis**: Main control button
- **Real-Time Analysis Box**:
  - AI Prediction: "Confident", "Nervous", or "Neutral"
  - Overall Score with progress bar
  - Live indicator when active
- **Real-Time Tips**: Dynamic improvement suggestions
- **Performance Metrics**: 5 metrics with full-width display
  - Blink Rate
  - Head Stability
  - Eye Contact
  - Gesture Activity
  - Voice Volume
- **Session Summary** (after stopping):
  - Overall score
  - Most common prediction
  - Average scores across session
  - Detailed feedback by category
  - "How to Get Confident" guide (if score < 75%)
- **Analyzing State**: Shows spinner while processing summary

### Design

- **Liquid Glass Morphism**: Frosted glass effects with backdrop blur
- **Dark/Light Mode**: System preference + manual toggle (improved contrast)
- **Responsive Design**: Works on desktop, tablet, mobile
- **Real-Time Updates**: Live metrics refresh every 500ms (stops after recording)
- **Smooth Animations**: Transitions and hover effects

## 📈 Performance

- **Training Accuracy**: 92.7% (no-glasses), 85.4% (with-glasses)
- **Base Features**: 37 per frame (visual + audio)
- **With Hand Gestures**: ~50+ features (includes hand motion, finger count, etc.)
- **Dataset**: 13,367+ frames (RAVDESS + optional hand gesture datasets)
- **Latency**: Real-time analysis (< 100ms per frame)
- **Prediction Override**: Rule-based boost to "Confident" when performance is excellent (70%+ overall with 2+ good metrics)

## 🔒 Privacy

Everything runs **locally** on the user's computer:
- Frontend captures frames in browser (never uploaded)
- Backend processes video/audio on-device
- No data sent to cloud
- Perfect for sensitive content like interviews/presentations

## 🐛 Troubleshooting

**Backend not starting?**
```bash
# Check dependencies
pip install -r requirements.txt

# Verify models exist
ls models/*.pkl

# Check shape_predictor file
ls backend/shape_predictor_68_face_landmarks.dat
```

**Frontend not loading?**
```bash
cd frontend
rm -rf .next node_modules
npm install
npm run dev
```

**Camera not working?**
- Browser will ask permission - click "Allow"
- Make sure no other app is using webcam
- Try different browser (Chrome recommended)

**Port conflicts?**
- Backend: Change port in `backend/app.py`
- Frontend: Change `-p 3001` in `frontend/package.json`

**Models not found?**
```bash
cd backend
python process_ravdess.py  # Extract features
python train_svm.py        # Train models
```

## 📚 API Endpoints

### Health Check
```bash
curl http://localhost:5000/
```
Returns: System status and loaded models

### Start Recording
```bash
curl -X POST http://localhost:5000/start
```
Returns: `{"status": "started"}`

### Send Frame (Frontend → Backend)
```bash
curl -X POST http://localhost:5000/process_frame \
  -H "Content-Type: application/json" \
  -d '{"frame": "data:image/jpeg;base64,..."}'
```

### Get Prediction (Poll every 500ms)
```bash
curl http://localhost:5000/process
```
Returns:
```json
{
  "prediction": "Confident",
  "scores": {
    "blink": 65,
    "head_stability": 85,
    "gesture_activity": 70,
    "eye_contact": 90,
    "voice": 75
  }
}
```

### Stop Recording
```bash
curl -X POST http://localhost:5000/stop
```
Returns: `{"status": "stopped"}`

## 🤏 Hand Gesture Dataset Integration (Optional)

**Note:** MirrorMe works perfectly **without** a hand gesture dataset! Real-time hand detection is active using classical CV. Adding a dataset improves training accuracy but isn't required.

### Quick Start with HGM-4 (Smallest Dataset)

1. **Download HGM-4** (~50-100 MB):
   - Visit: https://data.mendeley.com/datasets/jzy8zngkbg/8
   - Create free Mendeley account if needed
   - Download and extract to `data/small_gestures/HGM-1.0/`

2. **Process the Dataset**:
   ```bash
   cd backend
   python process_hand_gestures.py hgm4 ../data/small_gestures/HGM-1.0 Front_CAM
   ```
   This creates `data/hand_gesture_features.npz`

3. **Merge with RAVDESS**:
   ```bash
   python merge_datasets.py
   ```
   Creates `data/combined_features.npz`

4. **Retrain Models**:
   ```bash
   python train_svm.py
   ```
   Models will use the combined dataset automatically!

### Supported Datasets

| Dataset | Size | Format | Gestures | Best For |
|---------|------|--------|----------|----------|
| **HGM-4** | ~50-100 MB | Images | 26 (A-Z) | **Quick testing** ⭐ |
| **LeapGestRecog** | ~1.2 GB | Images | 10 | Balanced size/quality |
| **HaGRID** | ~716 GB | Images | 18 | Production quality |
| **IPN Hand** | ~35 GB | Videos | 13 | Video analysis |

### Processing Different Dataset Types

```bash
cd backend

# For HGM-4 (multi-camera dataset)
python process_hand_gestures.py hgm4 ../data/small_gestures/HGM-1.0 Front_CAM

# For HaGRID (image-based)
python process_hand_gestures.py hagrid ../data/hagrid

# For IPN Hand (video-based)
python process_hand_gestures.py video ../data/ipn_hand
```

### Dataset Links

- **HGM-4**: https://data.mendeley.com/datasets/jzy8zngkbg/8 (Smallest, recommended for testing)
- **LeapGestRecog**: https://www.kaggle.com/datasets/gti-upm/leapgestrecog (Medium size, good balance)
- **HaGRID**: https://github.com/hukenovs/hagrid (Best quality, large)
- **IPN Hand**: https://gibranbenitez.github.io/IPN_Hand/ (Video format)

## 👁️ Computer Vision Techniques Used

MirrorMe uses **33+ core computer vision techniques** for real-time analysis:

### **Face Analysis (4 techniques)**
- **dlib HOG Face Detector** - Histogram of Oriented Gradients for face detection
- **Haar Cascade** - Viola-Jones algorithm (fallback)
- **68-Point Facial Landmarks** - dlib shape predictor
- **PnP + RANSAC** - 3D head pose estimation (yaw, pitch, roll)

### **Feature Detection (6 techniques)**
- **Harris Corner Detection** - Corner points for gaze/smile
- **Shi-Tomasi Corner Detection** - Improved corner tracking
- **SIFT** - Scale-invariant feature detection
- **ORB** - Fast binary feature detector (real-time optimized)
- **Hough Circle Transform** - Iris detection
- **Local Binary Pattern (LBP)** - Texture analysis for glasses

### **Hand Gesture Analysis (10 techniques)**
- **HSV Skin Detection** - Color-based hand segmentation
- **Canny Edge Detection** - Hand edge detection
- **Contour Analysis** - Hand region detection
- **Convex Hull** - Hand shape approximation
- **Convexity Defects** - Finger valley detection
- **Optical Flow (Farneback)** - Motion estimation
- **Image Moments** - Shape descriptors (m00, m01, m10)
- **Blob Detection** - Hand region counting
- **Color Histograms** - Color diversity analysis
- **Circularity** - Shape compactness measure

### **Preprocessing (4 techniques)**
- **Color Space Conversion** - BGR↔Grayscale↔HSV
- **Otsu Thresholding** - Automatic binary threshold
- **Canny Edge Detection** - Multi-stage edge detection
- **CLAHE** - Contrast-limited adaptive histogram equalization

### **Additional CV Techniques**
- **Euclidean Distance** - Geometric calculations (EAR, MAR)
- **Morphological Operations** - Erosion, dilation, bitwise operations
- **Histogram Analysis** - Feature and color histograms

**Total: 33+ Core CV Techniques** - All using classical computer vision, no deep learning!

For detailed implementation, see `backend/extract_features.py`.

## 📄 License

This project is part of a CV course assignment. Feel free to use and modify.

## 🙏 Acknowledgments

- RAVDESS dataset for training data
- dlib for facial landmarks
- OpenCV for computer vision
- Next.js team for amazing framework

---

**Made with ❤️ for overcoming public speaking anxiety**

*One practice session at a time, privately and intelligently.*
