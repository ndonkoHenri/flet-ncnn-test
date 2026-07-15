# ncnn on Flet — on-device inference demo

A tiny Flet app that proves the native Python packages **`ncnn`**, **`opencv-python`**
and **`numpy`** (built for mobile by [mobile-forge](https://github.com/flet-dev/mobile-forge)
and published to `pypi.flet.dev`) load their compiled libraries and run **real neural-network
inference on-device** under Flet 0.86 — fully offline, no PyTorch/ultralytics on the device.

This is the demo for [flet-dev/flet discussion #6581](https://github.com/flet-dev/flet/discussions/6581):
the earlier bug where ncnn's native `.so` was stripped out of the APK is fixed.

## What it does

- Imports `ncnn` / `cv2` / `numpy` / `PIL` and shows their versions — this alone proves the
  native libraries load (that was the failure).
- Loads a bundled **SqueezeNet v1.1** ncnn model and classifies a bundled sample image,
  showing the top-5 ImageNet predictions.

Object detection (e.g. YOLOv8n) uses the identical mechanism — export the model to ncnn on
desktop (`yolo export model=yolov8n.pt format=ncnn`), bundle the `.param`/`.bin`, and swap the
pre/post-processing.

### Model asset

`src/model/` holds the canonical ncnn SqueezeNet example, vendored so the app runs offline:

- `squeezenet_v1.1.param`, `squeezenet_v1.1.bin` — from [nihui/ncnn-assets](https://github.com/nihui/ncnn-assets) (the source the `ncnn` package's own `model_zoo` uses)
- `synset_words.txt` — the 1000 ImageNet class labels, from [Tencent/ncnn](https://github.com/Tencent/ncnn/blob/master/examples/synset_words.txt)
- `sample.jpg` — a sample image to classify

## Build (CI)

Pushing to the repo runs `.github/workflows/build.yml`, which builds the **`apk`** and
**`ios-simulator`** artifacts. Download the `apk-build-artifact`, install the `.apk` on a device,
and open the app — it should show the package versions and immediately classify the sample image.

## Run locally (desktop)

```bash
uv run flet run
```

## Build locally

```bash
flet build apk -v      # Android
flet build ipa -v      # iOS
```
