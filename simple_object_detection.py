#!/usr/bin/env python3
"""
simple_object_detection.py

The starting point for this project. Loads a pre-trained MobileNet SSD model
and runs it on the Pi camera feed in real time. Nothing fancy here — just the
core detection loop so it's easy to follow what's actually happening.

Tested on: Raspberry Pi 5 + Pi Camera Module 3
Model: MobileNet SSD v1 quantized (detect.tflite, trained on COCO)

Before running, make sure you have the model files:
    mkdir -p models/
    # place detect.tflite and coco_labels.txt inside models/

Install dependencies (inside your venv):
    pip install picamera2 opencv-python numpy tflite-runtime

Controls:
    q  quit
    s  save current frame as a JPEG
"""

import cv2
import numpy as np
from picamera2 import Picamera2
import time

# tflite-runtime is the lightweight option (~2 MB) and what we want on the Pi.
# Full tensorflow also works but it's 500 MB and overkill for just running inference.
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite


MODEL_PATH           = "models/detect.tflite"
LABELS_PATH          = "models/coco_labels.txt"
CONFIDENCE_THRESHOLD = 0.5   # tweak this if you get too many false positives (raise) or miss things (lower)


def load_labels(path):
    with open(path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    # some label files ship with "???" as a dummy first line — just drop it
    if labels[0] == "???":
        labels.pop(0)
    return labels


def main():
    print("Loading labels...")
    labels = load_labels(LABELS_PATH)
    print(f"  {len(labels)} classes loaded")

    print("Loading model...")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # the model tells us what size it expects — no need to hard-code it
    input_shape = input_details[0]["shape"]
    model_h     = input_shape[1]
    model_w     = input_shape[2]
    print(f"  Model expects {model_w}x{model_h} input ({input_details[0]['dtype'].__name__})")

    print("Starting camera...")
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
    picam2.start()
    time.sleep(2)   # give the sensor a moment to settle and auto-expose

    print("\nRunning — press q to quit, s to save a frame\n")

    frame_count     = 0
    detection_count = 0

    try:
        while True:
            frame = picam2.capture_array()   # comes out as RGB from picamera2

            # the model wants BGR (OpenCV convention), resized to its input size
            input_img  = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (model_w, model_h))
            input_data = np.expand_dims(input_img, axis=0)

            # quantized (INT8) models want uint8, float models want 0-1 floats
            if input_details[0]["dtype"] == np.uint8:
                input_data = input_data.astype(np.uint8)
            else:
                input_data = input_data.astype(np.float32) / 255.0

            # this is the actual inference call — everything above is just prep
            t0 = time.time()
            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()
            inference_ms = (time.time() - t0) * 1000

            # MobileNet SSD outputs 4 tensors: boxes, classes, scores, num_detections
            boxes          = interpreter.get_tensor(output_details[0]["index"])[0]
            classes        = interpreter.get_tensor(output_details[1]["index"])[0]
            scores         = interpreter.get_tensor(output_details[2]["index"])[0]
            num_detections = int(interpreter.get_tensor(output_details[3]["index"])[0])

            img_h, img_w = frame.shape[:2]
            found = 0

            for i in range(num_detections):
                if scores[i] < CONFIDENCE_THRESHOLD:
                    continue

                # boxes come back as normalised [ymin, xmin, ymax, xmax] (0.0-1.0)
                ymin, xmin, ymax, xmax = boxes[i]
                left   = int(xmin * img_w)
                top    = int(ymin * img_h)
                right  = int(xmax * img_w)
                bottom = int(ymax * img_h)

                class_id   = int(classes[i])
                label      = labels[class_id] if class_id < len(labels) else f"id:{class_id}"
                confidence = scores[i] * 100

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                text      = f"{label}: {confidence:.1f}%"
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                # small filled rect so the text is readable over any background
                cv2.rectangle(frame, (left, top - 20), (left + text_size[0], top), (0, 255, 0), -1)
                cv2.putText(frame, text, (left, top - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                found += 1

            fps = 1000 / inference_ms if inference_ms > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}  inference: {inference_ms:.0f}ms  objects: {found}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Object Detection", frame)

            frame_count     += 1
            detection_count += found

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                fname = f"frame_{int(time.time())}.jpg"
                cv2.imwrite(fname, frame)
                print(f"Saved {fname}")

    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    picam2.stop()

    print(f"\nDone. {frame_count} frames, {detection_count} total detections")
    if frame_count:
        print(f"Avg detections/frame: {detection_count / frame_count:.2f}")


if __name__ == "__main__":
    main()
