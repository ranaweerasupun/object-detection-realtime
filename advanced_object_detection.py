#!/usr/bin/env python3
"""
advanced_object_detection.py

Built on top of simple_object_detection.py. The core detection logic is the
same, but this version is restructured into classes and adds a few things that
make it more practical to use day-to-day:

"""

import cv2
import numpy as np
from picamera2 import Picamera2
import time
from collections import deque

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite


MODEL_PATH           = "models/detect.tflite"
LABELS_PATH          = "models/coco_labels.txt"
CONFIDENCE_THRESHOLD = 0.6   # slightly higher than the simple version
CAMERA_WIDTH         = 640
CAMERA_HEIGHT        = 480
DETECTION_INTERVAL   = 2     # run inference every N frames — rest of the time we reuse the last result

# one colour per class (cycles if there are more than 6 classes in view)
COLORS = [
    (0, 255, 0),    # green
    (255, 80, 0),   # blue-ish
    (0, 80, 255),   # red-ish
    (255, 255, 0),  # cyan
    (255, 0, 200),  # magenta
    (0, 220, 255),  # yellow
]


class ObjectDetector:
    """Handles model loading and inference. Keeps track of FPS history."""

    def __init__(self, model_path, labels_path):
        with open(labels_path, "r") as f:
            self.labels = [line.strip() for line in f.readlines()]
        if self.labels[0] == "???":
            self.labels.pop(0)
        print(f"  {len(self.labels)} labels loaded")

        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        shape            = self.input_details[0]["shape"]
        self.model_h     = shape[1]
        self.model_w     = shape[2]
        print(f"  Model input: {self.model_w}x{self.model_h}")

        # keep the last 30 inference times so FPS doesn't flicker on screen
        self.fps_history     = deque(maxlen=30)
        self.last_detections = []

    def detect(self, frame, threshold):
        """
        Run inference on frame (RGB numpy array).
        Returns a list of detections and the inference time in ms.
        """
        t0 = time.time()

        resized    = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (self.model_w, self.model_h))
        input_data = np.expand_dims(resized, axis=0)

        if self.input_details[0]["dtype"] == np.uint8:
            input_data = input_data.astype(np.uint8)
        else:
            input_data = input_data.astype(np.float32) / 255.0

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        inference_ms = (time.time() - t0) * 1000
        self.fps_history.append(1000 / inference_ms if inference_ms > 0 else 0)

        boxes          = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        classes        = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
        scores         = self.interpreter.get_tensor(self.output_details[2]["index"])[0]
        num_detections = int(self.interpreter.get_tensor(self.output_details[3]["index"])[0])

        detections = []
        for i in range(num_detections):
            if scores[i] < threshold:
                continue
            cid = int(classes[i])
            detections.append({
                "box"        : tuple(boxes[i]),    # normalised (ymin, xmin, ymax, xmax)
                "class_id"   : cid,
                "class_name" : self.labels[cid] if cid < len(self.labels) else f"id:{cid}",
                "confidence" : float(scores[i]),
            })

        self.last_detections = detections
        return detections, inference_ms

    def avg_fps(self):
        return sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0.0

class Visualizer:
    """Draws boxes, labels, and the stats overlay onto frames."""

    def draw_detections(self, frame, detections):
        h, w = frame.shape[:2]
        for det in detections:
            ymin, xmin, ymax, xmax = det["box"]
            left, top, right, bottom = int(xmin*w), int(ymin*h), int(xmax*w), int(ymax*h)

            color = COLORS[det["class_id"] % len(COLORS)]
            label = f"{det['class_name']}: {det['confidence']*100:.1f}%"

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (left, top - lh - 8), (left + lw, top), color, -1)
            cv2.putText(frame, label, (left, top - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        return frame

    def draw_hud(self, frame, fps, inference_ms, num_objects, threshold):
        lines = [
            f"FPS: {fps:.1f}",
            f"Inference: {inference_ms:.0f} ms",
            f"Objects: {num_objects}",
            f"Threshold: {threshold:.0%}",
        ]
        y = 28
        for line in lines:
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y += 26
        return frame
    

def main():
    print("Advanced Object Detection")
    print("=" * 50)

    detector   = ObjectDetector(MODEL_PATH, LABELS_PATH)
    visualizer = Visualizer()

    # wrap in a list so the key handler can mutate it without a global
    threshold = [CONFIDENCE_THRESHOLD]

    print("Starting camera...")
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT)}
    ))
    picam2.start()
    time.sleep(2)

    print("\nRunning — q=quit  s=save  +/-=threshold\n")

    frame_count     = 0
    last_detections = []
    last_inf_ms     = 0.0
    class_totals    = {}   # running count per class for the end-of-session summary

    try:
        while True:
            frame = picam2.capture_array()

            if frame_count % DETECTION_INTERVAL == 0:
                last_detections, last_inf_ms = detector.detect(frame, threshold[0])
                for det in last_detections:
                    name = det["class_name"]
                    class_totals[name] = class_totals.get(name, 0) + 1

            visualizer.draw_detections(frame, last_detections)
            visualizer.draw_hud(frame, detector.avg_fps(), last_inf_ms,
                                len(last_detections), threshold[0])

            cv2.imshow("Advanced Object Detection", frame)
            frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                fname = f"frame_{int(time.time())}.jpg"
                cv2.imwrite(fname, frame)
                print(f"Saved {fname}")
            elif key in (ord("+"), ord("=")):
                threshold[0] = min(0.95, threshold[0] + 0.05)
                print(f"Threshold -> {threshold[0]:.0%}")
            elif key in (ord("-"), ord("_")):
                threshold[0] = max(0.10, threshold[0] - 0.05)
                print(f"Threshold -> {threshold[0]:.0%}")

    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    picam2.stop()

    print(f"\nFrames: {frame_count}  |  Avg FPS: {detector.avg_fps():.1f}")
    print("\nDetections by class:")
    if class_totals:
        for name, count in sorted(class_totals.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name:<20} {count}")
    else:
        print("  nothing detected")


if __name__ == "__main__":
    main()
