# Office Monitoring System

A Flask-based web application for monitoring office workspaces using computer vision and YOLO object detection. The system tracks people's presence in different working areas and measures time spent in each zone.

## Features

- Real-time person detection and tracking using YOLOv8
- Multiple working area zones monitoring
- Time tracking for each defined workspace
- Web interface for video upload and processing
- Processed video download capability
- Arabic UI interface

## Prerequisites

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLO
- Flask
- CUDA-capable GPU (optional, for better performance)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd office-monitoring
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the YOLOv8 model:
The system will automatically download the YOLOv8n model on first run.

## Project Structure

```
├── app.py              # Flask web application
├── main.py            # Core processing logic
├── utils.py           # Utility functions
├── templates/         # HTML templates
│   └── index.html    # Upload interface
├── uploads/          # Uploaded videos storage
└── output_video/     # Processed videos storage
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open a web browser and navigate to `http://localhost:5000`

3. Upload a video through the web interface

4. The system will process the video and:
   - Track people in defined working areas
   - Calculate time spent in each zone
   - Generate a processed video with visualizations

5. Download the processed video using the provided link

## Working Areas

The system monitors 6 predefined working areas:
- Area 1-6: Different workspace zones defined by polygon coordinates
- Each area tracks:
  - Person presence
  - Time spent by each person
  - Entry/exit events

## Technical Details

### Detection and Tracking
- Uses YOLOv8 for person detection
- Implements object tracking for consistent ID assignment
- Processes frames at specified intervals

### Time Tracking
- Maintains entry/exit timestamps for each tracked person
- Calculates cumulative time spent in each zone
- Updates in real-time during video processing

### Visualization
- Color-coded polygons for working areas
- Person detection boxes with ID labels
- Time overlay showing duration in each zone

## License

[Specify your license here]

## Contributors

- Team Morgan

## Acknowledgments

- YOLOv8 by Ultralytics
- OpenCV community
- Flask framework