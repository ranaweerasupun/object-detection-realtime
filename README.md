# Real-time Object Detection on Raspberry Pi 5

Running real-time object detection on a Raspberry Pi 5 using a pre-trained MobileNet SSD model and TensorFlow Lite. No cloud, no internet connection needed during inference — everything runs locally on the Pi.

This is a learning project. I'm building it up in stages, so the scripts get progressively more capable. The dev log (`LOG.md`) tracks what I tried, what broke, and what actually worked.

---

## Hardware

- Raspberry Pi 5 (8 GB)
- Pi Camera Module 3
- Python 3.11 (Raspberry Pi OS Bookworm)

## Model

**MobileNet SSD v1** — quantized INT8, trained on the [COCO dataset](https://cocodataset.org/) (80 object classes).

Downloaded from Google's TFLite model zoo:
```
https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
```

The quantized version runs noticeably faster on the Pi than the float32 one, with minimal accuracy drop.

---

## Project structure

```
object-detection-realtime/
├── models/
│   ├── detect.tflite       # model weights (download separately, see below)
│   └── coco_labels.txt     # 80 COCO class names
│
├── simple_object_detection.py   just a simple detection loop
│
├── LOG.md                  # dev log
└── README.md
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/ranaweerasupun/object-detection-realtime.git
cd object-detection-realtime
```

**2. Create a virtual environment**

> Important: install everything inside the venv. Mixing venv and system packages caused version conflicts during development.

```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install picamera2 opencv-python numpy tflite-runtime
```

**4. Download the model**
```bash
mkdir -p models
cd models

wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip

# rename to what the scripts expect
mv detect.tflite detect.tflite   # usually already named this
```

Then create `models/coco_labels.txt` with one class name per line (80 lines, starting with `person`). A copy is included in this repo.

---

## Scripts

### `simple_object_detection.py`
The bare minimum. One file, no classes, easy to read top to bottom. Good starting point if you want to understand how the pipeline works.

```bash
python simple_object_detection.py
```

**Controls:** `q` quit &nbsp;|&nbsp; `s` save frame

## Performance (Pi 5, MobileNet SSD v1 quantized)

| Resolution | Threads | Approx FPS |
|------------|---------|------------|
| 300×300    | 1       | ~8–10 fps  |
| 300×300    | 4       | ~12–15 fps |
| 640×480    | 4       | ~6–8 fps   |

These are rough numbers — actual FPS depends on scene complexity and what else is running on the Pi.

---


## Resources

- [TensorFlow Lite guide](https://www.tensorflow.org/lite/guide)
- [COCO dataset](https://cocodataset.org/)
- [Picamera2 docs](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf)
- [MobileNet paper](https://arxiv.org/abs/1704.04861)
