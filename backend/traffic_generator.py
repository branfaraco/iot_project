

from __future__ import annotations

import asyncio
import os
import h5py
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from shared.utils import data_reduction as dr


from dotenv import load_dotenv
load_dotenv()


def _load_hdf5_frames(data_dir: str) -> List[dict]:
    """Load traffic frames from HDF5 files in chronological order.

    This helper scans a directory for ``.h5`` files, reads the
    dataset named ``array`` from each file and extracts the frames
    corresponding to the simulation window (08:20–18:20 inclusive).
    Frames are concatenated across files to form a single list.
    """
    if not os.path.isdir(data_dir):
        raise RuntimeError(
            f"[traffic_generator] TRAFFIC_DATA_PATH directory not found: {data_dir}"
        )


    frames: List[dict] = []

    # Collect all .h5 files and sort them for reproducibility
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(".h5")]
    files.sort()
    
    if not files:
        raise RuntimeError(
            f"[traffic_generator] No .h5 files found in directory: {data_dir}"
        )

    print(f"[traffic_generator] Found {len(files)} .h5 files in {data_dir}")

    # Define the daily time window
    window_start = datetime(2019, 6, 1, 8, 20)
    window_end = datetime(2019, 6, 1, 18, 20)
    delta = timedelta(minutes=5)

    for fname in files:
        path = os.path.join(data_dir, fname)
        try:
            with h5py.File(path, "r") as f:
                arr = f["array"][()]  # (T, 495, 436, 8)
            print(f"[traffic_generator] Loaded file {fname} with shape {arr.shape}")

            # Apply the temporal crop only; spatial cropping and normalisation
            # are now handled in the backend.  This reduces the number of
            # frames per day but preserves the full spatial extent and raw
            # integer values.
            arr = dr.temporal_crop(arr, dr.T_START, dr.T_END)
            print(f"[traffic_generator] After temporal crop {fname}: {arr.shape}")

        except Exception as e:
            raise RuntimeError(
                f"[traffic_generator] Failed to load file {fname}: {e}"
            ) from e

        T, H, W, C = arr.shape
        frames_per_day = int(
            ((window_end - window_start).total_seconds() / 60) / 5) + 1
        for day_start in range(0, T, frames_per_day):
            day_end = min(day_start + frames_per_day, T)
            for i in range(day_start, day_end):
                frames.append(
                    {
                        "frame": arr[i],
                        "timestamp": (window_start + delta * (i - day_start)).isoformat() + "Z",
                        "file": fname,
                    }
                )

    if not frames:
        raise RuntimeError(
            f"[traffic_generator] No frames extracted from files in {data_dir}"
        )

    print(
        f"[traffic_generator] Loaded {len(frames)} frames in total; "
        f"first frame shape = {frames[0]['frame'].shape}"
    )
    return frames


def build_dataset() -> List[dict]:
    """Construct the list of frames for this generator.

    Uses the directory specified by TRAFFIC_DATA_PATH. No synthetic
    data is generated; any problem raises a RuntimeError.
    """
    path = os.environ.get("TRAFFIC_DATA_PATH")
    if not path:
        raise RuntimeError(
            "[traffic_generator] TRAFFIC_DATA_PATH not set. "
            "Set it to the directory containing your BERLIN HDF5 files, e.g. "
            "TRAFFIC_DATA_PATH=./hugging_face/BERLIN_reduced/data"
        )

    print(f"[traffic_generator] Using TRAFFIC_DATA_PATH={path}")
    frames = _load_hdf5_frames(path)
    # _load_hdf5_frames already raises if something goes wrong
    return frames


app = FastAPI(title="Traffic Generator Service")

DATASET: List[dict] = []  # define at module level


@app.on_event("startup")
async def startup_event() -> None:
    """Initialise the in-memory dataset on startup."""
    global DATASET
    DATASET = build_dataset()
    print(
        f"[traffic_generator] Startup complete. "
        f"Dataset contains {len(DATASET)} frames, "
        f"first frame shape = {DATASET[0]['frame'].shape}"
    )


@app.websocket("/traffic_stream")
async def traffic_stream(websocket: WebSocket) -> None:
    """WebSocket endpoint that streams raw traffic frames.

    The generator supports a simple control protocol described in the
    module docstring.  It maintains an internal pointer into the
    dataset and can pause/resume streaming as instructed by the
    backend.  When the end of the data is reached the generator
    sends an ``end_of_data`` message and stops emitting frames until
    the next ``start`` command resets the state.
    """
    await websocket.accept()
    print("[traffic_generator] Client connected to /traffic_stream")
    paused: bool = False
    running: bool = False
    speed: float = 1.0  # seconds per 5‑minute step
    index: int = 0
    dataset = DATASET

    async def send_frames() -> None:
        nonlocal index, running, paused, speed, dataset
        while True:
            # Wait until a run is started
            if not running:
                await asyncio.sleep(0.1)
                continue
            # Skip sending whilst paused
            if paused:
                await asyncio.sleep(0.1)
                continue
            # If we've reached the end of the dataset emit end_of_data once
            if index >= len(dataset):
                await websocket.send_json({"type": "end_of_data"})
                print("[traffic_generator] Sent end_of_data; stopping run")
                running = False
                # Do not reset index here; wait for a fresh start
                await asyncio.sleep(0.1)
                continue
            # Construct and send the next frame message
            entry = dataset[index]
            payload = {
                "type": "frame",
                "file": entry.get("file", ""),
                "index": index,
                "timestamp": entry.get("timestamp", ""),
                "frame": entry.get("frame").tolist(),
            }
            await websocket.send_json(payload)
            print(
                f"[traffic_generator] Sent frame index={index}, timestamp={payload['timestamp']}"
            )
            index += 1
            await asyncio.sleep(float(speed))

    async def receive_controls() -> None:
        nonlocal running, paused, speed, index
        while True:
            try:
                msg = await websocket.receive_json()
            except WebSocketDisconnect:
                # Client disconnected, stop loops
                running = False
                print("[traffic_generator] Client disconnected from /traffic_stream")
                break
            if not isinstance(msg, dict):
                continue
            print(f"[traffic_generator] Received control message: {msg}")
            mtype = msg.get("type")
            if mtype == "start":
                # Reset index and configure speed
                index = 0
                running = True
                paused = False
                speed = float(msg.get("speed", speed))
                print(
                    f"[traffic_generator] Start command: speed={speed}, resetting index"
                )
            elif mtype == "pause":
                paused = True
                print("[traffic_generator] Pause command received")
            elif mtype == "resume":
                paused = False
                print("[traffic_generator] Resume command received")
            elif mtype == "set_speed":
                # Preserve running/paused state
                new_speed = msg.get("speed")
                if new_speed is not None:
                    speed = float(new_speed)
                    print(
                        f"[traffic_generator] Set_speed command: new speed={speed}"
                    )
            elif mtype == "stop":
                # End current run and emit end_of_data
                running = False
                await websocket.send_json({"type": "end_of_data"})
                print("[traffic_generator] Stop command received; sent end_of_data")
            else:
                # Unknown control type; ignore
                print(f"[traffic_generator] Unknown control type: {mtype}")

    sender = asyncio.create_task(send_frames())
    receiver = asyncio.create_task(receive_controls())
    done, pending = await asyncio.wait(
        [sender, receiver], return_when=asyncio.FIRST_COMPLETED
    )
    # Cancel any pending tasks
    for task in pending:
        task.cancel()
