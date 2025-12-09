# 🚀 YOLOv8 Real-time Object Detection System

A professional, production-ready Python project for real-time object detection using **YOLOv8** and **OpenCV**. This system provides high-performance object detection with comprehensive features including logging, snapshots, and real-time visualization.

## ✨ Features

### 🎯 **Core Detection**
- **YOLOv8 pre-trained model** (yolov8n.pt for fast demo)
- **80 COCO classes** (person, phone, chair, laptop, bottle, etc.)
- **Real-time processing** with webcam or video files
- **Confidence scoring** for each detection

### 📹 **Input Options**
- **Webcam feed** (default)
- **Video file support** (MP4, AVI, etc.)
- **Flexible source selection** via command line

### 🖼️ **Visual Output**
- **Bounding boxes** with class names and confidence scores
- **Real-time FPS counter**
- **Frame counter** and object count
- **Model information** display
- **Clean, professional interface**

### 📊 **Advanced Features**
- **Detection logging** to CSV files
- **Snapshot saving** of frames with detections
- **Interactive controls** for confidence adjustment
- **Comprehensive error handling**

## 🛠️ Requirements

- **Python 3.8+**
- **OpenCV 4.8+**
- **PyTorch 1.9+**
- **Ultralytics 8.0+**
- **Webcam** (for real-time detection)

## 📦 Installation

### 1. **Clone or Download**
```bash
git clone <repository-url>
cd object-detection-system
```

### 2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Verify Installation**
```bash
python -c "import ultralytics; import cv2; print('✅ All dependencies installed!')"
```

## 🚀 Quick Start

### **Basic Usage (Webcam)**
```bash
python object_detection.py
```

### **Advanced Usage**
```bash
# Use different YOLO model
python object_detection.py --model yolov8s.pt

# Process video file
python object_detection.py --source video.mp4

# Adjust confidence threshold
python object_detection.py --confidence 0.7

# Enable automatic snapshots
python object_detection.py --snapshots

# Combine options
python object_detection.py --model yolov8m.pt --source video.mp4 --confidence 0.6
```

## 🎮 Controls

| Key | Action |
|-----|--------|
| **'q'** | Quit the application |
| **'s'** | Save snapshot of current frame |
| **'+'** | Increase confidence threshold |
| **'-'** | Decrease confidence threshold |

## 📁 Project Structure

```
├── object_detection.py      # Main detection system
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── output/                 # Detection logs (auto-created)
│   └── detections_*.csv   # Detection data exports
└── snapshots/              # Saved frames (auto-created)
    └── detection_*.jpg     # Frame snapshots
```

## 🔧 Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `yolov8n.pt` | Path to YOLO model file |
| `--source` | str | `0` | Video source (0=webcam, or file path) |
| `--confidence` | float | `0.5` | Confidence threshold (0.1-0.9) |
| `--snapshots` | flag | `False` | Enable automatic snapshot saving |

## 📊 Supported Object Classes

The system detects **80+ objects** including:

- **People**: person
- **Electronics**: cell phone, laptop, tv, keyboard, mouse
- **Furniture**: chair, couch, bed, dining table
- **Food**: bottle, cup, bowl, apple, banana
- **Vehicles**: car, bicycle, motorcycle, bus
- **Animals**: cat, dog, bird, horse
- **And many more...**

## 🎯 Model Options

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| **yolov8n.pt** | 6.3 MB | ⚡ Fast | Good | Demo, Real-time |
| **yolov8s.pt** | 22.5 MB | 🚀 Medium | Better | Production |
| **yolov8m.pt** | 52.2 MB | 🐌 Slow | Best | High Accuracy |
| **yolov8l.pt** | 87.7 MB | 🐌 Slow | Excellent | Research |
| **yolov8x.pt** | 136.2 MB | 🐌 Slow | Outstanding | Best Quality |

## 📈 Performance Metrics

- **Real-time processing**: 20-60 FPS (depending on model and hardware)
- **Memory usage**: 2-8 GB RAM (depending on model size)
- **CPU optimized**: Works on CPU (GPU recommended for best performance)
- **Scalable**: Easy to switch between different YOLO models

## 🔍 Customization

### **Change Model**
```python
detector = YOLOv8Detector(model_path="yolov8s.pt")
```

### **Adjust Confidence**
```python
detector.confidence_threshold = 0.7
```

### **Add Custom Classes**
Edit the `class_names` list in the `YOLOv8Detector` class.

## 🚨 Troubleshooting

### **Model Download Issues**
```bash
# YOLOv8 models are automatically downloaded on first use
# If issues occur, manually download:
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### **Webcam Issues**
- Ensure webcam is not in use by other applications
- Try different camera index: `--source 1`
- Check webcam permissions

### **Performance Issues**
- Use smaller model: `--model yolov8n.pt`
- Increase confidence threshold: `--confidence 0.7`
- Close other applications
- Use GPU if available

### **Dependency Issues**
```bash
# Reinstall dependencies
pip uninstall ultralytics opencv-python torch
pip install -r requirements.txt
```

## 📊 Output Files

### **Detection Logs (CSV)**
- **Location**: `output/detections_YYYYMMDD_HHMMSS.csv`
- **Columns**: frame_no, class_name, confidence, timestamp
- **Format**: Comma-separated values for analysis

### **Snapshots (Images)**
- **Location**: `snapshots/detection_YYYYMMDD_HHMMSS_mmm.jpg`
- **Content**: Frames with detected objects
- **Trigger**: Manual save ('s' key) or automatic (--snapshots flag)

## 🔬 Advanced Usage

### **Batch Processing**
```bash
# Process multiple video files
for video in *.mp4; do
    python object_detection.py --source "$video" --confidence 0.6
done
```

### **Integration with Other Systems**
```python
from object_detection import YOLOv8Detector

# Initialize detector
detector = YOLOv8Detector(model_path="yolov8s.pt")

# Process single frame
frame = cv2.imread("image.jpg")
detections = detector.detect_objects(frame)
```

### **Custom Confidence Thresholds**
```bash
# High precision (fewer false positives)
python object_detection.py --confidence 0.8

# High recall (more detections)
python object_detection.py --confidence 0.3
```

## 🤝 Contributing

Feel free to contribute by:
- **Adding new features** (object tracking, recording)
- **Improving performance** (optimization, GPU support)
- **Enhancing documentation** (examples, tutorials)
- **Bug fixes** and error handling improvements

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Ultralytics** for YOLOv8 implementation
- **OpenCV** for computer vision capabilities
- **COCO dataset** for object classes
- **PyTorch** for deep learning framework

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Try with default settings first

---

**🎯 Ready to detect objects in real-time!** 

Run `python object_detection.py` and start detecting objects with professional-grade accuracy! 🚀
