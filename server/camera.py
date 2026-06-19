"""
Recycling Lab Tycoon — Camera Capture Module
=============================================
Provides a thread-safe, async-friendly wrapper around an OpenCV VideoCapture
that grabs frames at a configurable FPS and exposes them to the inference loop.
"""

import asyncio
import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraCapture:
    """
    Continuously captures frames from a webcam in a background thread.

    The latest frame is always available via :meth:`get_frame` (non-blocking)
    or :meth:`get_frame_async` (awaitable).

    Parameters
    ----------
    camera_index : int
        OpenCV camera index (default ``0``).
    target_fps : int
        Target capture rate in frames-per-second (default ``10``).
    width : int | None
        Optional resolution width hint.
    height : int | None
        Optional resolution height hint.
    """

    def __init__(
        self,
        camera_index: int = 0,
        target_fps: int = 10,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        self.camera_index = camera_index
        self.target_fps = target_fps
        self._interval = 1.0 / target_fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._width = width
        self._height = height

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Open the camera and begin capturing in a background thread."""
        if self._running:
            logger.warning("Camera is already running.")
            return

        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {self.camera_index}. "
                "Check that a webcam is connected."
            )

        # Apply resolution hints if given
        if self._width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        if self._height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            "Camera %d opened  (%d × %d) @ target %d FPS",
            self.camera_index,
            actual_w,
            actual_h,
            self.target_fps,
        )

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the capture thread and release the camera."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        if self._cap and self._cap.isOpened():
            self._cap.release()
            logger.info("Camera %d released.", self.camera_index)
        self._cap = None
        self._frame = None

    # ── Frame access ──────────────────────────────────────────────────

    def get_frame(self) -> Optional[np.ndarray]:
        """Return the most recent frame (BGR, uint8) or ``None``."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    async def get_frame_async(self) -> Optional[np.ndarray]:
        """Awaitable version of :meth:`get_frame`."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_frame)

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Internal capture loop ─────────────────────────────────────────

    def _capture_loop(self) -> None:
        """Background thread that grabs frames at the target FPS."""
        logger.info("Capture thread started.")
        while self._running:
            t0 = time.perf_counter()

            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame
            else:
                logger.warning("Frame capture failed — retrying …")

            # Throttle to target FPS
            elapsed = time.perf_counter() - t0
            sleep_time = self._interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("Capture thread stopped.")
