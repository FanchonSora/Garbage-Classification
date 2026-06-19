"""
Recycling Lab Tycoon — Async WebSocket Server
==============================================
Core server that ties camera capture, model inference, and ensemble voting
together, broadcasting results to connected Unity clients every ~100 ms.
"""

import asyncio
import json
import logging
import signal
import time
from typing import Set

import websockets
from websockets.asyncio.server import Server, ServerConnection

from camera import CameraCapture
from ensemble import EnsembleResult, majority_vote
from model_loader import ModelLoader

logger = logging.getLogger(__name__)


class RecyclingLabServer:
    """
    WebSocket server for the Recycling Lab Tycoon AI backend.

    Workflow (every cycle):
      1. Grab the latest camera frame.
      2. Run inference on all 5 models.
      3. Apply majority-vote ensemble.
      4. Broadcast the JSON result to every connected Unity client.

    Parameters
    ----------
    model_loader : ModelLoader
        Pre-initialised model loader with all weights loaded.
    camera : CameraCapture
        Camera capture instance (already started).
    host : str
        Bind address (default ``"localhost"``).
    port : int
        Bind port (default ``8765``).
    inference_interval : float
        Seconds between inference cycles (default ``0.1`` → 10 FPS).
    """

    def __init__(
        self,
        model_loader: ModelLoader,
        camera: CameraCapture,
        host: str = "localhost",
        port: int = 8765,
        inference_interval: float = 0.1,
    ):
        self.model_loader = model_loader
        self.camera = camera
        self.host = host
        self.port = port
        self.inference_interval = inference_interval

        self._clients: Set[ServerConnection] = set()
        self._server: Server | None = None
        self._running = False

        # Stats
        self._total_cycles = 0
        self._total_inference_time = 0.0

    # ── Client management ─────────────────────────────────────────────

    async def _register(self, ws: ServerConnection) -> None:
        self._clients.add(ws)
        logger.info(
            "Client connected: %s  (total: %d)",
            ws.remote_address,
            len(self._clients),
        )

    async def _unregister(self, ws: ServerConnection) -> None:
        self._clients.discard(ws)
        logger.info(
            "Client disconnected: %s  (total: %d)",
            ws.remote_address,
            len(self._clients),
        )

    async def _handler(self, ws: ServerConnection) -> None:
        """Handle a single WebSocket connection lifetime."""
        await self._register(ws)
        try:
            # Keep connection alive; we push data, client just listens.
            # But we also listen for any incoming messages (e.g. pings).
            async for message in ws:
                # Unity may send control messages in the future
                logger.debug("Received from client: %s", message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self._unregister(ws)

    # ── Broadcast ─────────────────────────────────────────────────────

    async def _broadcast(self, payload: str) -> None:
        """Send a JSON string to all connected clients."""
        if not self._clients:
            return

        # Use websockets.broadcast for efficient fan-out
        websockets.broadcast(self._clients, payload)

    # ── Inference loop ────────────────────────────────────────────────

    async def _inference_loop(self) -> None:
        """
        Main loop: grab frame → infer → vote → broadcast.
        Runs until ``self._running`` is set to ``False``.
        """
        logger.info("Inference loop started  (interval=%.0f ms)", self.inference_interval * 1000)
        loop = asyncio.get_event_loop()

        while self._running:
            cycle_start = time.perf_counter()

            frame = await self.camera.get_frame_async()
            if frame is None:
                await asyncio.sleep(self.inference_interval)
                continue

            # Run inference in a thread-pool so we don't block the event loop
            t0 = time.perf_counter()
            raw_predictions = await loop.run_in_executor(
                None, self.model_loader.predict_all, frame
            )
            inference_time = time.perf_counter() - t0

            # Ensemble vote
            predictions = [(cls, conf) for cls, conf, _ in raw_predictions]
            result: EnsembleResult = majority_vote(predictions)

            # Build JSON packet
            packet = result.to_json()
            packet["inference_ms"] = round(inference_time * 1000, 1)
            payload = json.dumps(packet)

            # Broadcast
            await self._broadcast(payload)

            # Stats
            self._total_cycles += 1
            self._total_inference_time += inference_time
            if self._total_cycles % 50 == 0:
                avg_ms = (self._total_inference_time / self._total_cycles) * 1000
                logger.info(
                    "Stats: %d cycles | avg inference %.0f ms | clients %d | "
                    "last: %s (%.2f, %s)",
                    self._total_cycles,
                    avg_ms,
                    len(self._clients),
                    result.detected_item,
                    result.confidence,
                    result.method,
                )

            # Log every prediction for debugging (at DEBUG level)
            logger.debug(
                "→ %s  conf=%.3f  method=%s  votes=%s  infer=%dms",
                result.detected_item,
                result.confidence,
                result.method,
                result.all_votes,
                inference_time * 1000,
            )

            # Maintain target interval
            elapsed = time.perf_counter() - cycle_start
            sleep = self.inference_interval - elapsed
            if sleep > 0:
                await asyncio.sleep(sleep)

    # ── Server lifecycle ──────────────────────────────────────────────

    async def start(self) -> None:
        """Start the WebSocket server and the inference loop."""
        self._running = True

        self._server = await websockets.serve(
            self._handler,
            self.host,
            self.port,
        )
        logger.info(
            "WebSocket server listening on ws://%s:%d",
            self.host,
            self.port,
        )

        # Run inference loop concurrently
        await self._inference_loop()

    async def stop(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down server …")
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        # Close all client connections
        for ws in list(self._clients):
            await ws.close()

        logger.info("Server stopped.")
