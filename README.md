# People Counter with Camera and YOLOv5

This application uses a USB camera to capture images, detects people using YOLOv5, and provides the count via a FastAPI server.

## Features
- Capture photo from USB camera
- Count number of people using YOLOv5 object detection
- Serve the count via FastAPI endpoint

## Installation
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Ensure the YOLOv5 model is available at `/app/pydev_demo/models/yolov5s_672x672_nv12.bin`

3. Ensure `libpostprocess.so` is available at `/usr/lib/libpostprocess.so`

## Usage
Run the server:
```
python main.py
```

Access the endpoint:
```
GET http://localhost:8000/count_people
```

Response:
```json
{
  "people_count": 3
}
```

## Notes
- Requires a USB camera connected.
- Uses custom hobot_dnn library for inference.