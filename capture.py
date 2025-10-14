import threading
import time
import os
from typing import Optional, Tuple

import cv2


class BackgroundCapture:
    """Continuously reads from a V4L2 video device on a background thread and keeps the latest frame.

    This class gates initial frames (warmup) and can reject obviously bad frames (very dark/blank).
    Call get_frame() to retrieve the most recent good frame with an optional timeout.
    """

    def __init__(self,
                 device_path: str,
                 preferred_formats: Tuple[Tuple[str, int], ...] = (
                     ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
                     ('YUYV', cv2.VideoWriter_fourcc(*'YUYV')),
                 ),
                 preferred_resolutions: Tuple[Tuple[int, int], ...] = (
                     (1920, 1080), (1280, 720), (640, 480)
                 ),
                 warmup_frames: int = 30,
                 target_fps: Optional[float] = None):
        self.device_path = device_path
        self.preferred_formats = preferred_formats
        self.preferred_resolutions = preferred_resolutions
        self.warmup_frames = max(0, int(warmup_frames))
        self.target_delay = (1.0 / target_fps) if target_fps and target_fps > 0 else None

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._latest = None  # type: Optional[Tuple[float, any]]

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="BackgroundCapture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None
        self._ready.clear()

    def is_ready(self) -> bool:
        return self._ready.is_set()

    def get_frame(self, timeout_sec: float = 0.2):
        """Return the most recent good frame.

        If not ready yet, wait up to timeout_sec for readiness. Returns None if unavailable.
        """
        if not self._ready.wait(timeout=timeout_sec):
            return None
        with self._lock:
            if self._latest is None:
                return None
            # Return a copy to avoid race with writer
            ts, frame = self._latest
            return frame.copy()

    # Internal helpers
    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        if not os.path.exists(self.device_path):
            return None
        cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.device_path, cv2.CAP_ANY)
        if not cap.isOpened():
            return None

        for fmt_name, fmt_code in self.preferred_formats:
            for width, height in self.preferred_resolutions:
                cap.set(cv2.CAP_PROP_FOURCC, fmt_code)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.read()
                ok, frame = cap.read()
                if ok and frame is not None:
                    return cap
        cap.release()
        return None

    @staticmethod
    def _frame_is_good(frame) -> bool:
        # Drop obviously blank/dark frames
        if frame is None:
            return False
        mean = frame.mean()
        if mean < 2.0:  # near-black
            return False
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._cap is None:
                self._ready.clear()
                self._cap = self._open_capture()
                if self._cap is None:
                    time.sleep(0.2)
                    continue

                # Warmup
                for _ in range(self.warmup_frames):
                    self._cap.read()

            start_time = time.time()
            ok, frame = self._cap.read()
            if not ok or frame is None:
                # Reopen on failure
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
                continue

            if self._frame_is_good(frame):
                with self._lock:
                    self._latest = (time.time(), frame)
                self._ready.set()

            if self.target_delay is not None:
                elapsed = time.time() - start_time
                remaining = self.target_delay - elapsed
                if remaining > 0:
                    time.sleep(remaining)


