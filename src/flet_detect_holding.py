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
    
    # Target path adjustment for local script execution vs Flet bundling
    asset_dir = os.path.dirname(__file__) if "ANDROID_ARGUMENT" in os.environ else "assets"
    
    param_path = os.path.join(asset_dir, "model.ncnn.param")
    bin_path = os.path.join(asset_dir, "model.ncnn.bin")
    img_path = os.path.join(asset_dir, "man.png")

    # COCO Class names for yolov8n
    class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    # Initialize NCNN Network Architecture
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False  # Set to True on Android devices for GPU boost
    
    if net.load_param(param_path) != 0:
        raise FileNotFoundError(f"Could not load structure from: {param_path}")
    if net.load_model(bin_path) != 0:
        raise FileNotFoundError(f"Could not load weights from: {bin_path}")

    # UI Widgets
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
            
            # Map pixels to NCNN format scaled to 640x640
            ncnn_img = ncnn.Mat.from_pixels_resize(
                bgr_img, 
                ncnn.Mat.PixelType.PIXEL_BGR2RGB, 
                orig_w, orig_h, 640, 640
            )

            # Preprocessing input image pixels normalization
            mean_vals = [0.0, 0.0, 0.0]
            norm_vals = [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]
            ncnn_img.substract_mean_normalize(mean_vals, norm_vals)

            # Forward pass inference extraction
            ex = net.create_extractor()
            ex.input("in0", ncnn_img)
            mat_out = ncnn.Mat()
            ex.extract("out0", mat_out)
            
            out_array = np.array(mat_out)  # Matrix shape configuration
            
            cx = out_array[0, :]
            cy = out_array[1, :]
            nw = out_array[2, :]
            nh = out_array[3, :]
            classes_conf = out_array[4:, :]
            
            best_class_indices = np.argmax(classes_conf, axis=0)
            best_confidences = np.max(classes_conf, axis=0)
            
            conf_threshold = 0.25
            valid_mask = best_confidences > conf_threshold
            
            # Center coordinates conversion to corner boundaries
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
                    
                    inds = np.where(iou <= nms_threshold)[0]
                    order = order[inds + 1]
            
            # --- Rendering Bounding Boxes & Identifying Held Items ---
            output_string = ""
            if len(keep_indices) == 0:
                output_string = "No objects detected."
            else:
                person_boxes = []
                other_detections = []
                
                # Split detected structures into people layers and object layers
                for idx in keep_indices:
                    class_id = class_ids[idx]
                    name = class_names[class_id]
                    conf = round(float(scores[idx]), 2)
                    
                    # Convert canvas values back to native file frame lengths
                    bx1 = int(boxes[idx, 0] * orig_w / 640.0)
                    by1 = int(boxes[idx, 1] * orig_h / 640.0)
                    bx2 = int(boxes[idx, 2] * orig_w / 640.0)
                    by2 = int(boxes[idx, 3] * orig_h / 640.0)
                    
                    box_coords = (bx1, by1, bx2, by2)
                    
                    if name == "person":
                        person_boxes.append(box_coords)
                        # Render person identification canvas borders in Blue
                        cv2.rectangle(bgr_img, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
                    else:
                        other_detections.append((name, conf, box_coords))

                # Spatial Intersection Check Loop
                held_items = []
                for name, conf, (ox1, oy1, ox2, oy2) in other_detections:
                    is_held = False
                    
                    for (px1, py1, px2, py2) in person_boxes:
                        # Determine center coordinate of individual items
                        cx_item = (ox1 + ox2) / 2
                        cy_item = (oy1 + oy2) / 2
                        
                        # Validate if center balances inside user boundaries
                        if px1 <= cx_item <= px2 and py1 <= cy_item <= py2:
                            is_held = True
                            break
                    
                    color = (0, 255, 0) if is_held else (0, 165, 255)  # Green for held items, Orange for others
                    status = " [HELD]" if is_held else ""
                    
                    output_string += f"Detected: {name} | Confidence: {conf}{status}\n"
                    if is_held:
                        held_items.append(name)
                    
                    # Paint items rectangles
                    cv2.rectangle(bgr_img, (ox1, oy1), (ox2, oy2), color, 2)
                    label = f"{name} {conf}{status}"
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(bgr_img, (ox1, oy1 - text_h - 4), (ox1 + text_w, oy1), color, -1)
                    cv2.putText(bgr_img, label, (ox1, oy1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                # Format interface header strings
                if person_boxes:
                    holding_summary = f"He is holding: {', '.join(held_items)}" if held_items else "Man is not holding anything recognized."
                    output_string = f"{holding_summary}\n\n" + output_string
                else:
                    output_string = "No person detected in the image.\n\n" + output_string

            # Convert BGR array into base64 stream string mapping
            _, buffer = cv2.imencode('.png', bgr_img)
            b64_string = base64.b64encode(buffer).decode('utf-8')
            
            display_image.src_base64 = b64_string
            result_text.value = output_string.strip()
            
        except Exception as ex_err:
            result_text.value = f"Runtime Error: {str(ex_err)}"
        
        page.update()

    page.add(
        ft.AppBar(title=ft.Text(" NCNN neural network inference Analysis")),
        ft.Container(height=10),
        display_image,
        ft.Container(height=10),
        ft.Button("Run NCNN Inference", on_click=run_inference),
        ft.Container(height=10),
        result_text
    )

ft.run(main)
