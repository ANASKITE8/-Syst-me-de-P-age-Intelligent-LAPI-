🚗 Vehicle Detection System - Step 1 of License Plate Recognition Pipeline
Author: Anas Elhassouni
Date: October 28, 2025
Version: 1.0.0

📋 Overview
This is Step 1 of a 3-step License Plate Recognition Pipeline:

Vehicle Detection (This Module) ✅
License Plate Detection (Next Team)
OCR Text Recognition (Final Team)

This module detects and tracks vehicles in video footage using YOLOv11, preparing data for the license plate detection step.

🎯 Features

✅ Real-time vehicle detection (cars, trucks, buses, motorcycles, bicycles, trains)
✅ Vehicle tracking with unique IDs
✅ Professional annotated video output
✅ JSON data export for pipeline integration
✅ Cropped vehicle images for next processing step
✅ Detection statistics and analytics


📁 Project Structure
vehicle_detection/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── main.py                      # Standalone script with UI
├── vehicle_detector.py          # Modular pipeline version
├── test_videos/
│   └── 4.mp4                   # Sample test video
├── outputs/
│   ├── output_professional.mp4  # Annotated video
│   ├── detections.json          # Detection data for next step
│   ├── vehicles/                # Cropped vehicle images
│   └── frames/                  # (Optional) Individual frames
└── models/
    └── yolo11s.pt              # YOLO model (auto-downloaded)

🚀 Quick Start
Prerequisites

Python 3.8 or higher
Windows/Linux/Mac
At least 4GB RAM
(Optional) GPU for faster processing

Installation

Clone or download this project
Install dependencies:

bash   pip install -r requirements.txt

Run the detection:

bash   # Option 1: Professional UI version
   python main.py
   
   # Option 2: Pipeline module
   python vehicle_detector.py
First Run

The YOLO model (yolo11s.pt) will be automatically downloaded (~20MB)
This only happens once


📖 Usage
Option 1: Standalone with Professional UI (main.py)
bashpython main.py
Controls:

SPACE - Pause/Resume
ESC - Exit
R - Reset counts

Features:

Real-time dashboard
Performance metrics
Visual effects
Color-coded vehicles

Option 2: Pipeline Module (vehicle_detector.py)
pythonfrom vehicle_detector import VehicleDetector

# Initialize
detector = VehicleDetector(
    model_path='models/yolo11s.pt',
    output_dir='outputs'
)

# Process video
stats = detector.process_video(
    video_path='test_videos/4.mp4',
    save_frames=False,
    frame_skip=1
)

# Get results
print(stats)

📤 Outputs
1. Annotated Video (outputs/output_professional.mp4)

Visual demonstration with bounding boxes
Vehicle IDs and classifications
Confidence scores

2. Detection Data (outputs/detections.json)
Format:
json{
  "metadata": {
    "total_detections": 1250,
    "unique_vehicles": 45,
    "timestamp": "2025-10-28T10:30:00"
  },
  "detections": [
    {
      "frame_number": 1,
      "track_id": 1,
      "vehicle_type": "car",
      "confidence": 0.95,
      "bbox": {
        "x1": 100,
        "y1": 200,
        "x2": 300,
        "y2": 400,
        "width": 200,
        "height": 200
      },
      "center": {
        "x": 200,
        "y": 300
      },
      "crop_path": "outputs/vehicles/vehicle_000001_0001.jpg",
      "timestamp": "2025-10-28T10:30:01"
    }
  ]
}
3. Vehicle Crops (outputs/vehicles/)

Individual images of each detected vehicle
Naming format: vehicle_[frame]_[track_id].jpg
Used by next team for license plate detection


🔗 Integration Guide for Next Team (License Plate Detection)
Input Data You'll Receive:

JSON file (detections.json) with:

Frame numbers
Vehicle bounding boxes
Vehicle types
Track IDs


Cropped vehicle images in outputs/vehicles/
(Optional) Original video file

How to Use:
pythonimport json

# Load detection data
with open('outputs/detections.json', 'r') as f:
    data = json.load(f)

# Process each detection
for detection in data['detections']:
    vehicle_image = cv2.imread(detection['crop_path'])
    # Your license plate detection code here
    # ...
Recommended Approach:

Load the JSON file
For each vehicle detection:

Load the cropped image
Run license plate detection
Extract plate region


Save plate crops for OCR team


⚙️ Configuration
Detection Parameters
Edit vehicle_detector.py:
python# Vehicle classes to detect
vehicle_classes = {
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    6: 'train',
    7: 'truck'
}

# Confidence threshold
if conf > 0.5:  # Adjust this value (0.0 to 1.0)
    # Save detection
Performance Options
python# Process every Nth frame (speed up processing)
frame_skip = 1  # 1 = all frames, 2 = every other frame

# Save individual frames
save_frames = False  # Set to True if needed

📊 Statistics & Analytics
The system provides:

Total frames processed
Number of detections
Unique vehicles tracked
Vehicle type distribution
Processing speed (FPS)


🐛 Troubleshooting
Issue: "No module named 'ultralytics'"
Solution:
bashpip install ultralytics
Issue: Video not found
Solution:

Ensure video is in test_videos/ folder
Check video filename matches code
Use absolute path if needed

Issue: Slow processing
Solutions:

Use GPU if available
Increase frame_skip parameter
Reduce video resolution
Use a lighter YOLO model (yolo11n.pt)

Issue: Low detection accuracy
Solutions:

Adjust confidence threshold
Use larger model (yolo11m.pt or yolo11l.pt)
Improve video quality
Adjust lighting conditions


🔧 Advanced Usage
Using Different YOLO Models
python# Faster but less accurate
detector = VehicleDetector(model_path='models/yolo11n.pt')

# Slower but more accurate
detector = VehicleDetector(model_path='models/yolo11l.pt')
Processing Multiple Videos
pythonvideos = ['video1.mp4', 'video2.mp4', 'video3.mp4']

for video in videos:
    detector = VehicleDetector()
    detector.process_video(video)
    detector.save_detections_json(f'{video}_detections.json')
Custom Output Directory
pythondetector = VehicleDetector(output_dir='results/experiment_1')

📝 Notes for Team

Model file: YOLO11s provides good balance of speed/accuracy
JSON format: Standardized for easy parsing by next team
Vehicle crops: High-quality images for plate detection
Track IDs: Consistent across frames for tracking
Timestamps: For synchronization if needed


🤝 Contact & Support
Questions about this module?

Check the code comments
Review the JSON output format
Test with provided sample video

For the next team:

Use detections.json as your input
Vehicle crops are ready in outputs/vehicles/
Frame numbers match original video


📜 License
This project is part of the License Plate Recognition Pipeline.
For educational and research purposes.

🔄 Version History

v1.0.0 (2025-10-28): Initial release

Vehicle detection and tracking
JSON export for pipeline
Professional UI option
Complete documentation




📚 References

Ultralytics YOLOv11 Documentation
OpenCV Documentation
COCO Dataset Classes



Happy Detecting! 🚗💨
