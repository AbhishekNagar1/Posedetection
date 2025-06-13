# AI Fitness Pose Estimation App 🧘‍♂️📸

<div align="center">
  <a href="https://license-instructions.netlify.app/" target="_blank">
    <img src="https://img.shields.io/badge/🚨-READ%20BEFORE%20FORKING-red?style=for-the-badge&labelColor=darkred" alt="Read Before Forking">
  </a>
</div>

A computer vision-based fitness application that uses **pose estimation** to evaluate human posture during workouts. This project leverages OpenCV and deep learning models to detect, analyze, and provide feedback on user movements — making fitness smarter and more interactive.

---

## 📁 Project Structure

```
├── fitnessapp/              # Core app logic and server
├── PoseImages/              # Stored pose-related images
├── PoseVideos/              # Stored pose-related videos
├── __pycache__/             # Cached Python bytecode
├── .idea/                   # IDE config files
├── PoseEstimationMin.py     # Lightweight pose estimation script
├── PoseModule.py            # Pose utility functions & logic
├── animate.py               # Pose animation visualizer
├── imgPose.py               # Image-based pose analysis
├── multirealpose.py         # Multi-user real-time pose tracking
├── realPose.py              # Real-time single-user pose tracking
├── testcode.py              # Experimental / test scripts
├── yolov8n.pt               # YOLOv8n model weights
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## 🚀 Features

- 🔍 **Real-time pose detection** with OpenCV and MediaPipe
- 🤖 **YOLOv8n integration** for accurate human detection
- 📈 **Visual graphs** for movement tracking and analysis
- 🧠 **Improved model accuracy** for fitness-specific poses
- 👥 **Multi-user support** for group workouts
- 🎬 **Animation visualizer** for pose feedback
- 🖼️ **Image and video processing** capabilities
- 🧪 **Test scripts** for development and debugging

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **OpenCV** - Computer vision library
- **MediaPipe** - Pose estimation framework
- **Ultralytics YOLOv8** - Object detection model
- **Matplotlib** - Data visualization
- **NumPy** - Numerical computing
- **Flask/FastAPI** - Web framework (if applicable)

---

## 🏃‍♂️ Getting Started

### Prerequisites

- Python 3.8 or higher
- Webcam or camera device
- GPU recommended for better performance

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/AbhishekNagar1/ai-fitness-pose-estimation.git
cd ai-fitness-pose-estimation
```

2. **Create virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download model weights** (if not included)

The YOLOv8n model weights should be automatically downloaded on first run. If not, you can download them manually:

```bash
# YOLOv8n weights will be downloaded automatically
# No manual download required
```

---

## 🎯 Usage

### Real-time Pose Tracking (Single User)

```bash
python realPose.py
```

### Multi-user Real-time Tracking

```bash
python multirealpose.py
```

### Image-based Pose Analysis

```bash
python imgPose.py
```

### Pose Animation Visualizer

```bash
python animate.py
```

### Minimal Pose Estimation

```bash
python PoseEstimationMin.py
```

---

## 📊 Key Components

### Core Files

- **`PoseModule.py`** - Main pose detection and analysis utilities
- **`realPose.py`** - Single-user real-time pose tracking
- **`multirealpose.py`** - Multi-user pose tracking system
- **`imgPose.py`** - Static image pose analysis
- **`animate.py`** - Pose animation and visualization
- **`PoseEstimationMin.py`** - Lightweight pose estimation implementation

### Data Directories

- **`PoseImages/`** - Sample images and processed results
- **`PoseVideos/`** - Video files for testing and demos
- **`fitnessapp/`** - Web application components (if applicable)

---

## 🎮 Features Overview

### Pose Detection Capabilities

- **33 body landmarks** detection using MediaPipe
- **Real-time processing** at 30+ FPS
- **Angle calculation** for joint movements
- **Pose classification** for different exercises
- **Movement tracking** and analysis

### Supported Exercises

- Push-ups
- Squats
- Planks
- Bicep curls
- Shoulder exercises
- Custom pose definitions

---

## 📸 Demo & Screenshots

*Add your demo videos, screenshots, or GIFs here to showcase the application in action.*

### Example Output

The application provides:
- Real-time pose overlay on video feed
- Joint angle measurements
- Exercise form feedback
- Movement tracking graphs
- Performance metrics

---

## 🔧 Configuration

### Camera Settings

Modify camera parameters in the respective Python files:

```python
# Camera resolution
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Detection confidence
DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE = 0.5
```

### Model Parameters

Adjust pose detection sensitivity and other parameters in `PoseModule.py`.

---

## 📝 API Reference

### PoseModule Class

```python
from PoseModule import PoseDetector

detector = PoseDetector()
# Methods available:
# - findPose(img)
# - findPosition(img)
# - findAngle(img, p1, p2, p3)
```

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📋 Requirements

Create a `requirements.txt` file with:

```
opencv-python>=4.5.0
mediapipe>=0.8.0
ultralytics>=8.0.0
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.7.0
```

---

## 🐛 Troubleshooting

### Common Issues

1. **Camera not detected**: Check camera permissions and connections
2. **Low FPS**: Reduce camera resolution or use GPU acceleration
3. **Model not loading**: Ensure YOLOv8 weights are properly downloaded
4. **ImportError**: Verify all dependencies are installed correctly

### Performance Optimization

- Use GPU acceleration if available
- Reduce camera resolution for better FPS
- Optimize detection confidence thresholds
- Close unnecessary applications while running

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Abhishek Nagar**

- GitHub: [@AbhishekNagar1](https://github.com/AbhishekNagar1)
- Website: https://abhishek-nagar-portfolio.netlify.app/

---

## 🙏 Acknowledgments

- MediaPipe team for the pose estimation framework
- Ultralytics for YOLOv8 implementation
- OpenCV community for computer vision tools
- Contributors and testers

---

## 🔮 Future Enhancements

- [ ] Web-based interface
- [ ] Mobile app development
- [ ] Advanced exercise recognition
- [ ] Performance analytics dashboard
- [ ] Cloud deployment
- [ ] Multi-language support

---

⚡ **"Built to move you, powered by code."**

*Making fitness smarter through computer vision and AI.*
