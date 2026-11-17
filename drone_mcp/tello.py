"""Driver for a real Ryze Tello EDU, through the djitellopy library.

UNTESTED until it meets a real drone. It implements the same interface as
SimDrone, so the safety gate, perception and MCP tools don't change.

The Tello has no GPS, so position is dead reckoning: the driver adds up the
moves it was told to make. Expect drift of a few centimetres per move; the
flight box should leave margin for it.

Setup:
  pip install djitellopy opencv-python
  Tello EDU on your home Wi-Fi (station mode) so this computer keeps internet.
"""
from __future__ import annotations

import time
from pathlib import Path

from .drone import DroneState, offset


class TelloDrone:
    def __init__(self, frames_dir: str = "frames", host: str | None = None):
        from djitellopy import Tello   # imported here so the simulator never needs it
        self.t = Tello(host=host) if host else Tello()
        self.t.connect()
        self.t.streamon()
        self.reader = self.t.get_frame_read()
        self.frames = Path(frames_dir)
        self.frames.mkdir(parents=True, exist_ok=True)
        self.state = DroneState(battery=float(self.t.get_battery()))
        self._t0 = time.monotonic()

    def _sync(self):
        self.state.clock = time.monotonic() - self._t0
        try:
            self.state.battery = float(self.t.get_battery())
            if self.state.flying:
                self.state.z = float(self.t.get_height())
        except Exception:
            pass   # keep the last reading rather than fail a command over telemetry

    def takeoff(self):
        self.t.takeoff()
        self.state.flying = True
        self._sync()

    def land(self):
        self.t.land()
        self.state.flying = False
        self.state.z = 0.0
        self._sync()

    def move(self, direction: str, cm: int):
        getattr(self.t, f"move_{direction}")(int(cm))
        dx, dy, dz = offset(self.state.heading, direction, cm)
        self.state.x += dx
        self.state.y += dy
        self.state.z += dz
        self._sync()

    def rotate(self, degrees: int):
        if degrees >= 0:
            self.t.rotate_clockwise(int(degrees))
        else:
            self.t.rotate_counter_clockwise(int(-degrees))
        self.state.heading = (self.state.heading + degrees) % 360
        self._sync()

    def hover(self, seconds: float):
        # the Tello holds position on its own while the verifier works, so the
        # time has already passed; just refresh telemetry
        self._sync()

    def emergency(self):
        self.t.emergency()
        self.state.flying = False
        self._sync()

    def capture(self, tag: str = "") -> str:
        import cv2
        frame = self.reader.frame
        h, w = frame.shape[:2]
        if w > 1024:   # smaller frames are faster to check and still clear enough
            frame = cv2.resize(frame, (1024, int(h * 1024 / w)))
        path = self.frames / f"{time.strftime('%H%M%S')}-{int(time.monotonic() * 1000) % 1000:03d}-{tag or 'frame'}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return str(path)

    # the simulator answers these from its world; a real drone has no world model
    def visible(self):
        return []

    def frame_for(self, obj):
        return self.capture("candidate")
