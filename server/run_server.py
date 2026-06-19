#!/usr/bin/env python3
"""
Recycling Lab Tycoon — Server Entry Point
==========================================
Launch the AI backend WebSocket server.

Usage
-----
    python run_server.py                           # defaults
    python run_server.py --port 8765 --camera 0    # explicit
    python run_server.py --device cuda             # GPU inference
    python run_server.py --fps 5                   # slower cycle

The server will:
  1. Load all 5 garbage-classification models.
  2. Open the webcam and start capturing at the target FPS.
  3. Listen for Unity WebSocket connections on ws://host:port.
  4. Continuously run ensemble inference and broadcast results.

Press Ctrl+C to stop gracefully.
"""

import argparse
import asyncio
import logging
import os
import signal
import sys

# Ensure the server package is importable when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera import CameraCapture
from model_loader import ModelLoader
from websocket_server import RecyclingLabServer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recycling Lab Tycoon — AI WebSocket Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--weights-dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "models", "models-weight",
        ),
        help="Path to directory containing .pth weight files.",
    )
    p.add_argument("--host", type=str, default="localhost", help="Bind address.")
    p.add_argument("--port", type=int, default=8765, help="WebSocket port.")
    p.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    p.add_argument("--fps", type=int, default=10, help="Target inference FPS.")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device ('cpu' or 'cuda'). Auto-detected if omitted.",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    p.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the real-time OpenCV camera preview GUI window.",
    )
    return p.parse_args()


def setup_logging(level: str) -> None:
    fmt = "%(asctime)s │ %(levelname)-7s │ %(name)-24s │ %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=getattr(logging, level), format=fmt, datefmt=datefmt)
    # Quieten noisy libraries
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


async def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    logger = logging.getLogger("run_server")
    logger.info("=" * 60)
    logger.info("  RECYCLING LAB TYCOON — AI Backend Server")
    logger.info("=" * 60)

    # ── 1. Load models ────────────────────────────────────────────────
    weights_dir = os.path.abspath(args.weights_dir)
    logger.info("Weights directory: %s", weights_dir)

    model_loader = ModelLoader(weights_dir=weights_dir, device=args.device)

    # ── 2. Start camera ───────────────────────────────────────────────
    camera = CameraCapture(camera_index=args.camera, target_fps=args.fps)
    camera.start()

    # ── 3. Start WebSocket server ─────────────────────────────────────
    server = RecyclingLabServer(
        model_loader=model_loader,
        camera=camera,
        host=args.host,
        port=args.port,
        inference_interval=1.0 / args.fps,
        show_preview=not args.no_preview,
    )

    # Graceful shutdown on Ctrl+C
    loop = asyncio.get_event_loop()

    def _shutdown():
        logger.info("Interrupt received — stopping …")
        asyncio.ensure_future(server.stop())
        camera.stop()

    # Handle Ctrl+C
    try:
        loop.add_signal_handler(signal.SIGINT, _shutdown)
        loop.add_signal_handler(signal.SIGTERM, _shutdown)
    except NotImplementedError:
        # Windows doesn't support add_signal_handler
        pass

    try:
        await server.start()
    except KeyboardInterrupt:
        pass
    finally:
        await server.stop()
        camera.stop()
        logger.info("Goodbye!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
