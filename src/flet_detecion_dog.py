import os
import flet as ft
import ncnn
import cv2
import numpy as np
import base64

def main(page: ft.Page):
    page.title = "Flet NCNN Object Detection"
    page.theme_mode = ft.ThemeMode.DARK
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    
    asset_dir = os.path.dirname(__file__) if "ANDROID_ARGUMENT" in os.environ else "assets"
    
    param_path = os.path.join(asset_dir, "model.ncnn.param")
    bin_path = os.path.join(asset_dir, "model.ncnn.bin")
    img_path = os.path.join(asset_dir, "test-imag.png")

    class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    
    if net.load_param(param_path) != 0:
        raise FileNotFoundError(f"Could not load structure from: {param_path}")
    if net.load_model(bin_path) != 0:
        raise FileNotFoundError(f"Could not load weights from: {bin_path}")

    result_text = ft.Text("Ready", size=16, weight=ft.FontWeight.BOLD)
    
    display_image = ft.Image(
        src=img_path,
        width=400,
        height=400,
        fit=ft.BoxFit.CONTAIN,
        border_radius=ft.BorderRadius.all(10)
    )

    def run_inference(e):
        try:
            bgr_img = cv2.imread(img_path)
            if bgr_img is None:
                result_text.value = f"Error: Image not found at {img_path}"
                page.update()
                return

            orig_h, orig_w, _ = bgr_img.shape
            
            ncnn_img = ncnn.Mat.from_pixels_resize(
                bgr_img, 
                ncnn.Mat.PixelType.PIXEL_BGR2RGB, 
                orig_w, orig_h, 640, 640
            )

            mean_vals = [0.0, 0.0, 0.0]
            norm_vals = [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]
            ncnn_img.substract_mean_normalize(mean_vals, norm_vals)

            ex = net.create_extractor()
            ex.input("in0", ncnn_img)
            mat_out = ncnn.Mat()
            ex.extract("out0", mat_out)
            
            out_array = np.array(mat_out)
            
            cx = out_array[0, :]
            cy = out_array[1, :]
            nw = out_array[2, :]
            nh = out_array[3, :]
            classes_conf = out_array[4:, :]
            
            best_class_indices = np.argmax(classes_conf, axis=0)
            best_confidences = np.max(classes_conf, axis=0)
            
            conf_threshold = 0.25
            valid_mask = best_confidences > conf_threshold
            
            x1 = (cx - nw / 2)
            y1 = (cy - nh / 2)
            x2 = (cx + nw / 2)
            y2 = (cy + nh / 2)
            
            boxes = np.stack([x1, y1, x2, y2], axis=1)[valid_mask]
            scores = best_confidences[valid_mask]
            class_ids = best_class_indices[valid_mask]
            
            # --- Non-Maximum Suppression (NMS) ---
            nms_threshold = 0.45
            keep_indices = []
            
            if len(boxes) > 0:
                order = np.argsort(scores)[::-1]
                while order.size > 0:
                    i = order[0]
                    keep_indices.append(i)
                    if order.size == 1:
                        break
                    xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
                    yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
                    xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
                    yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
                    inter_w = np.maximum(0.0, xx2 - xx1)
                    inter_h = np.maximum(0.0, yy2 - yy1)
                    inter_area = inter_w * inter_h
                    area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
                    area_others = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
                    union_area = area_i + area_others - inter_area
                    iou = inter_area / union_area
                    
                    # FIX APPLIED HERE
                    inds = np.where(iou <= nms_threshold)[0]
                    order = order[inds + 1]
            
            # --- Rendering Bounding Boxes ---
            output_string = ""
            if len(keep_indices) == 0:
                output_string = "No objects detected."
            else:
                for idx in keep_indices:
                    class_id = class_ids[idx]
                    conf = round(float(scores[idx]), 2)
                    name = class_names[class_id]
                    output_string += f"Result: {name} | Confidence: {conf}\n"
                    
                    box_x1 = int(boxes[idx, 0] * orig_w / 640.0)
                    box_y1 = int(boxes[idx, 1] * orig_h / 640.0)
                    box_x2 = int(boxes[idx, 2] * orig_w / 640.0)
                    box_y2 = int(boxes[idx, 3] * orig_h / 640.0)
                    
                    cv2.rectangle(bgr_img, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)
                    
                    label = f"{name} {conf}"
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(bgr_img, (box_x1, box_y1 - text_h - 4), (box_x1 + text_w, box_y1), (0, 255, 0), -1)
                    cv2.putText(bgr_img, label, (box_x1, box_y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

            _, buffer = cv2.imencode('.png', bgr_img)
            b64_string = base64.b64encode(buffer).decode('utf-8')
            
            display_image.src_base64 = b64_string
            result_text.value = output_string.strip()
            
        except Exception as ex_err:
            result_text.value = f"Runtime Error: {str(ex_err)}"
        
        page.update()

    page.add(
        ft.AppBar(title=ft.Text("NCNN Bounding Box Render")),
        ft.Container(height=10),
        display_image,
        ft.Container(height=10),
        ft.Button("Run NCNN Inference", on_click=run_inference),
        ft.Container(height=10),
        result_text
    )

ft.run(main)
