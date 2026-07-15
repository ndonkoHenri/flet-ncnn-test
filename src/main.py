"""Proof that the native `ncnn` / `opencv-python` / `numpy` mobile wheels load and
run real neural-network inference on-device under Flet 0.86 — fully offline.

Loads a bundled SqueezeNet v1.1 ncnn model and classifies a bundled sample image.
"""

import os

import cv2
import flet as ft
import ncnn
import numpy as np
import PIL

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


def main(page: ft.Page):
    page.scroll = ft.ScrollMode.AUTO

    # Load the bundled SqueezeNet v1.1 model.
    net = ncnn.Net()
    net.load_param(os.path.join(MODEL_DIR, "squeezenet_v1.1.param"))
    net.load_model(os.path.join(MODEL_DIR, "squeezenet_v1.1.bin"))
    labels = (
        open(os.path.join(MODEL_DIR, "synset_words.txt"), encoding="utf-8")
        .read()
        .splitlines()
    )

    # Classify the bundled sample image.
    img_bytes = open(os.path.join(MODEL_DIR, "sample.jpg"), "rb").read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    mat_in = ncnn.Mat.from_pixels_resize(
        img, ncnn.Mat.PixelType.PIXEL_BGR, w, h, 227, 227
    )
    mat_in.substract_mean_normalize([104.0, 117.0, 123.0], [])
    ex = net.create_extractor()
    ex.input("data", mat_in)
    _, out = ex.extract("prob")
    probs = np.array(out)
    top5 = [
        (labels[i].split(" ", 1)[1], float(probs[i]))
        for i in probs.argsort()[-5:][::-1]
    ]

    page.add(
        ft.SafeArea(
            content=ft.Column(
                controls=[
                    ft.Text(
                        f"ncnn {ncnn.__version__} · opencv {cv2.__version__} · "
                        f"numpy {np.__version__} · pillow {PIL.__version__}",
                        size=12,
                        color=ft.Colors.GREY,
                    ),
                    ft.Image(
                        src=img_bytes,
                        width=260,
                        height=260,
                        fit=ft.BoxFit.CONTAIN,
                        border_radius=10,
                    ),
                    ft.Text("Top-5 predictions:", weight=ft.FontWeight.BOLD),
                    *[ft.Text(f"{p * 100:5.1f}%  {name}") for name, p in top5],
                ]
            )
        )
    )


ft.run(main)
