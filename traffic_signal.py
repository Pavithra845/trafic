from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLOv8 Model
model = YOLO("yolov8n.pt")  # Make sure this file is present in your working directory or use full path

# Initialize Video Capture
cap = cv2.VideoCapture(0)  # You can try cap = cv2.VideoCapture(1) if 0 doesn't work

if not cap.isOpened():
    print("❌ Error: Cannot access webcam. Please check camera or try another index.")
    exit()
else:
    print("✅ Webcam successfully opened.")

# Signal data
green_time = 30  # Default green light time in seconds
current_green_lane = "Lane 1"
lane1_time_left = green_time
lane2_time_left = green_time

def detect_objects(frame):
    results = model(frame)
    detected_objects = results[0].boxes
    vehicle_detected = False
    
    for obj in detected_objects:
        class_id = int(obj.cls[0])  # YOLO class ID
        if class_id in [2, 5, 7]:  # 2=car, 5=bus, 7=truck
            vehicle_detected = True
            break
    return vehicle_detected

@app.route('/')
def home():
    return render_template('index.html')  # Make sure index.html exists in /templates

@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_video():
    global lane1_time_left, lane2_time_left, current_green_lane

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame from camera")
            break

        # Detect objects
        vehicle_detected = detect_objects(frame)

        if vehicle_detected:
            if current_green_lane == "Lane 1":
                lane1_time_left -= 1
                if lane1_time_left <= 0:
                    current_green_lane = "Lane 2"
                    lane1_time_left = green_time
                    lane2_time_left = green_time
            else:
                lane2_time_left -= 1
                if lane2_time_left <= 0:
                    current_green_lane = "Lane 1"
                    lane1_time_left = green_time
                    lane2_time_left = green_time

        # Draw bounding boxes
        results = model(frame)
        frame = results[0].plot()

        # Encode frame
        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/signal_status')
def signal_status():
    return jsonify({
        'current_green_lane': current_green_lane,
        'green_time': green_time,
        'lane1_time_left': lane1_time_left,
        'lane2_time_left': lane2_time_left
    })

if __name__ == '__main__':
    app.run(debug=True)
