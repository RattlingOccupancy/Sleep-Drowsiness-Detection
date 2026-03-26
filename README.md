# 👁️ Enhanced Eye Tracking & Drowsiness Detection System


Implementation of real-time eye tracking and drowsiness monitoring using computer vision and deep learning. This system analyzes blink patterns and eye closure time (PERCLOS) to assess the alertness level of the user, providing a comprehensive dual-panel interface for real-time feedback.

---

## 🌟 Key Features

- **🚀 Real-Time Detection**: High-performance eye state analysis using a custom-trained Keras model.
- **📊 Comprehensive Monitoring**:
  - **Blink Rate**: Tracks blinks per minute (BPM) with a sliding window approach.
  - **PERCLOS**: Calculates the Percentage of Eye Closure over time to detect fatigue.
  - **Confidence Weighting**: Filters out false detections using model confidence scores.
- **🖥️ Dual-Panel UI**:
  - **System Panel**: Displays runtime, FPS, face/eye detection counts, and system status.
  - **Analysis Panel**: Real-time blink statistics, health assessment, and alertness state.
- **🔒 Robust Performance**: Built-in error handling for camera connectivity and model loading.
- **📝 Detailed Reporting**: Generate comprehensive session reports with clinical interpretations.

---

## 🛠️ Technology Stack

- **Language**: Python
- **Computer Vision**: OpenCV (Haar Cascades for face/eye localization)
- **Deep Learning**: TensorFlow / Keras (CNN for eye state classification)
- **Data Structures**: Collections (Deque for sliding window metrics)
- **Mathematics**: NumPy (Statistical analysis)

---

## 🗄️ Dataset — MRL Eye Dataset

The eye state classification model (`eye_open_close_model.keras`) was trained on the **MRL Eye Dataset** (Machine Learning in Real Life Eye Dataset), a large-scale open-source dataset specifically designed for eye-state and gaze research.

| Attribute | Details |
|-----------|---------|
| **Full Name** | MRL Eye Dataset (Machine Learning in Real Life) |
| **Images** | 84,898 infrared eye images |
| **Classes** | Open Eye / Closed Eye |
| **Subjects** | 37 individuals across diverse lighting & head-pose conditions |
| **Resolution** | Multiple resolutions available |
| **Source** | [mrl-eye.dbgraphics.com](http://mrl-eye.dbgraphics.com) |

### Why MRL?
- Captured under **real-world conditions** (varying luminance, gaze direction, and subject demographics), making it robust against environmental noise.
- Large enough to train a convolutional neural network that generalises well beyond lab settings.
- Widely cited in drowsiness-detection research, ensuring reproducibility and comparability with prior work.

---

## 📋 Prerequisites

Before running the application, ensure you have:
- A working webcam.
- Python 3.9 or higher installed.
- (Optional) NVIDIA GPU with CUDA drivers for faster model inference.

---

## ⚙️ Installation

1. **Clone the project** (or download the source):
   ```bash
   cd sdp
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Verify Assets**:
   Ensure the following files are present in the directory:
   - `main.py`
   - `models/eye_open_close_model.keras`
   - `requirements.txt`

---

## 🚀 Usage

Run the main application:
```bash
python main.py
```

### 🎮 Controls
| Key | Action |
|-----|--------|
| `q` | **Quit** the application safely |
| `r` | **Reset** all session statistics and calibration |
| `s` | **Save** a comprehensive text report of the current session |
| `c` | **Calibrate** detection sensitivity (adjusts blink threshold) |

---

## 📈 Understanding the Stats

### Alertness States
- **😊 ALERT**: Normal blink rate (12-20 BPM) and low PERCLOS.
- **😴 DROWSY**: Triggered when PERCLOS exceeds 70% or blink rate drops below 6 BPM for an extended period.

### Metrics Definitions
- **PERCLOS**: Percentage of time the eyes are more than 80% closed. It is the most reliable indicator of physiological drowsiness.
- **BPM (Blinks per Minute)**: Healthy adults typically blink 12-20 times per minute. Lower rates often indicate high focus or intense fatigue.
- **Confidence**: The probability score from the deep learning model regarding whether the eye is open or closed.

---

## 📁 File Structure

```text
sdp/
├── main.py                     # Core application logic and UI
├── requirements.txt            # Project dependencies
├── models/                     # Model architecture and weights
│   └── eye_open_close_model.keras/
│       ├── config.json
│       ├── metadata.json
│       └── model.weights.h5
└── README.md                   # You are here
```

---

## 💡 Best Practices

1. **Lighting**: Ensure your face is well-lit from the front. Avoid strong backlighting.
2. **Positioning**: Face the camera directly. The Haar Cascades are optimized for frontal face detection.
3. **Calibration**: Use the `c` key if the system is too sensitive or not sensitive enough to your blink patterns.

---

## ⚖️ License

This project is intended for research and educational purposes. Ensure compliance with local privacy laws when using camera-based monitoring systems.
