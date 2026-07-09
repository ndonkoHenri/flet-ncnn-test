
import ncnn
from ultralytics import YOLO

# Load the exported NCNN directory
net = ncnn.Net()
net.load_param("yolov8n_ncnn_model/yolov8n.param")
net.load_model("yolov8n_ncnn_model/yolov8n.bin")

# Load exported model via Ultralytics API for direct testing
model = YOLO("./yolov8n_ncnn_model")
results = model("./assets/test-img.png", task="detect", imgsz=640, conf=0.25)


# Process the first image result

for result in results:
    for box in result.boxes:
        class_id = int(box.cls.item())
        name = result.names[class_id]
        confidence = round(box.conf.item(), 2)
        print(f"Result: {name} | Confidence: {confidence}")

