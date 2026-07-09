"""
pip install ultralytics
pip install ncnn
"""
from ultralytics import YOLO

# 1. Load the model (Automatically downloads yolov8n.pt on the first run)
model = YOLO("yolov8n.pt")

# 2. Run inference on an image (Replace with a URL or local image path)
results = model.predict(source="https://hips.hearstapps.com/ghk.h-cdn.co/assets/17/30/dachshund.jpg?crop=1.00xw:0.668xh;0,0.260xh", conf=0.25)

# 3. View the prediction results
results[0].show()  # Opens a window displaying bounding boxes and labels
