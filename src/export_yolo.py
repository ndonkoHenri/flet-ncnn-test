"""
Step 1: Export Your Model to NCNN

Step 2: Set Up Your NCNN EnvironmentDepending on your target platform (Linux/Windows desktop, Raspberry Pi, or Android), you will need to set up the NCNN library:Linux/Windows: Install the library using your package manager or compile it from the Tencent NCNN GitHub Repository.

Step 3: Write the Inference CodeRunning NCNN inference requires loading the model, processing the raw image, and applying Non-Maximum Suppression (NMS).Using the Python API:pythonimport ncnn
from ultralytics import YOLO

# Load the exported NCNN directory
net = ncnn.Net()
net.load_param("yolov8n_ncnn_model/yolov8n.param")
net.load_model("yolov8n_ncnn_model/yolov8n.bin")

# Load exported model via Ultralytics API for direct testing
model = YOLO("./yolov8n_ncnn_model")
results = model("path/to/your/image.jpg")
"""
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="ncnn", imgsz=640)
