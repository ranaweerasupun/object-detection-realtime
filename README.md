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
├── simple_object_detection.py    # week 1 — bare-bones detection loop
├── advanced_object_detection.py  # week 2 — classes, frame skipping, colour coding
├── people_counter.py             # week 3 — occupancy counter with CSV logging
│
├── LOG.md                  # dev log — what I tried and what happened
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

---

### `advanced_object_detection.py`
Same model, better structure. Runs inference every N frames so the display FPS is smoother. Colour-coded boxes per class. You can adjust the confidence threshold on the fly with `+` / `-`.

```bash
python advanced_object_detection.py
```

**Controls:** `q` quit &nbsp;|&nbsp; `s` save &nbsp;|&nbsp; `+` / `-` adjust threshold

---

### `people_counter.py`
Detects only people (COCO class 0) and tracks current occupancy, session peak, and total. Logs a timestamped count to `people_count_log.csv` every frame.

```bash
python people_counter.py
```

**Controls:** `q` quit &nbsp;|&nbsp; `r` reset counters

---

## Performance (Pi 5, MobileNet SSD v1 quantized)

| Resolution | Threads | Approx FPS |
|------------|---------|------------|
| 300×300    | 1       | ~8–10 fps  |
| 300×300    | 4       | ~12–15 fps |
| 640×480    | 4       | ~6–8 fps   |

These are rough numbers — actual FPS depends on scene complexity and what else is running on the Pi.

---

## Troubleshooting

**Low FPS**
- Make sure you're using the quantized (INT8) model, not the float32 one
- Try `num_threads=4` in the `tflite.Interpreter()` call
- Increase `DETECTION_INTERVAL` in advanced_object_detection.py

**Objects not being detected**
- Check lighting first — this was the most common issue during development (see LOG.md)
- Lower `CONFIDENCE_THRESHOLD` (try 0.4)
- Make sure the object is one of the 80 COCO classes

**Camera not starting**
- Make sure the camera is enabled: `sudo raspi-config` → Interface Options → Camera
- Check the camera ribbon cable is seated properly

**Import errors**
- Make sure you're running inside the venv (`source venv/bin/activate`)
- Don't mix `pip install --break-system-packages` and venv installs

---

## Resources

- [TensorFlow Lite guide](https://www.tensorflow.org/lite/guide)
- [COCO dataset](https://cocodataset.org/)
- [Picamera2 docs](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf)
- [MobileNet paper](https://arxiv.org/abs/1704.04861)
