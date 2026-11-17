"""Tier-1 detector for real flights: YOLO-World, an open-vocabulary detector.

You give it the words to look for ("backpack") instead of retraining it, and it
runs fast enough for live video. UNTESTED here: it needs `pip install ultralytics`
and downloads its weights on first use.

Distance is a rough estimate from how tall the object looks in the frame, so
approach() moves in steps and re-checks rather than trusting one number.
"""
from __future__ import annotations

import math
import os

from .perception import Detection

FOV = 82.0                                   # Tello horizontal field of view
TYPICAL_HEIGHT_CM = float(os.environ.get("DRONE_TARGET_HEIGHT_CM", "45"))


class YoloWorldDetector:
    def __init__(self, weights: str | None = None, min_conf: float = 0.15):
        from ultralytics import YOLOWorld    # imported lazily: only real flights need it
        self.model = YOLOWorld(weights or os.environ.get("YOLO_WORLD_WEIGHTS", "yolov8s-world.pt"))
        self.min_conf = min_conf
        self._classes = None

    def detect(self, drone, query: str):
        # YOLO-World is told the object type; the verifier checks the details
        noun = query.strip().split()[-1] if query.strip() else query
        if self._classes != [noun]:
            self.model.set_classes([noun])
            self._classes = [noun]
        frame = drone.capture("detect")
        result = self.model.predict(frame, verbose=False)[0]
        h, w = result.orig_shape
        focal = (w / 2) / math.tan(math.radians(FOV / 2))
        out = []
        for box, conf in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
            if conf < self.min_conf:
                continue
            x0, y0, x1, y1 = box
            angle = ((x0 + x1) / 2 / w - 0.5) * FOV
            dist = TYPICAL_HEIGHT_CM * focal / max(1.0, y1 - y0)
            out.append(Detection(noun, round(conf, 2), round(angle, 1), round(dist), frame))
        return sorted(out, key=lambda d: -d.confidence)
