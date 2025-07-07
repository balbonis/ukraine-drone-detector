#KillUkraineDrone

from flask import Flask, Response
import cv2
from ultralytics import YOLO
from datetime import datetime

app = Flask(__name__)

# Load YOLOv8 model (choose yolov8n, yolov8s, yolov8m, yolov8l, yolov8x as needed)

#1. Default 
# #model = YOLO('yolov8s.pt')  # for all scans

#2. UKRAINE (Kamikaze Drone Images from Sky)
model = YOLO('C:/Users/drizz/Downloads/Dada/MyFlaskProject/ukraine_drone_detection/ukraine_drone_yolov8_model/weights/best.pt')

#Raybird3 (Drone Specific)
#model = YOLO('C:/Users/drizz/Downloads/Dada/MyFlaskProject/drone_detection/raybird3_yolov8_model/weights/best.pt')


# Start camera (0 = default webcam)
camera = cv2.VideoCapture(0)

# Open log file in append mode
log_file_path = 'UkrainDrone_detection_log.txt'
log_file = open(log_file_path, 'a')
def log_detection(message):
    log_file.write(message + '\n')
    log_file.flush()  # ensure it writes immediately
    print(message)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # YOLOv8 expects BGR (OpenCV default), so no need to convert
        results = model.predict(source=frame, stream=True)

        for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue  # no detections

                # You could inspect boxes for NaN (optional)
                if any((box.isnan().any() for box in r.boxes.xyxy)):
                    print("⚠ Skipping frame with NaN box coordinates")
                    continue

                # Check if drone (class 0) is detected
                classes = r.boxes.cls.int().tolist()
                if 0 in classes:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    message = f"[{timestamp}] Drone detected!"
                    log_detection(str(message))
                
                annotated_frame = r.plot()

                # Encode frame for streaming
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html>
    <head><title>Ukraine Drone Live Detector</title></head>
    <body>
        <h1>Ukraine Drone Live Detector</h1>
        <img src="/video_feed" width="800" height="600" class="center"/>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)