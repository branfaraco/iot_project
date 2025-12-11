from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Load .env 
from dotenv import load_dotenv
load_dotenv()


import pandas as pd


def _load_weather_records(data_dir: str) -> List[dict]:
    """Load cleaned weather data from CSV files.
    """

    test_dir = os.path.join(data_dir, "test")
    if not os.path.isdir(test_dir):
        raise RuntimeError(
            f"[weather_generator] Test directory not found: {test_dir}"
        )

    try:
        temp_path = os.path.join(test_dir, "temp.csv")
        precip_path = os.path.join(test_dir, "precip.csv")
        wind_path = os.path.join(test_dir, "wind.csv")

        df_temp = pd.read_csv(temp_path)
        df_prec = pd.read_csv(precip_path)
        df_wind = pd.read_csv(wind_path)
        print(
            f"[weather_generator] Loaded weather CSVs: "
            f"temp.csv ({len(df_temp)} rows), "
            f"precip.csv ({len(df_prec)} rows), "
            f"wind.csv ({len(df_wind)} rows)"
        )
    except Exception as e:
        raise RuntimeError(
            f"[weather_generator] Failed to load cleaned weather CSV files from {test_dir}: {e}"
        ) from e

    # Normalise timestamps and drop unused columns
    for df in (df_temp, df_prec, df_wind):
        df["MESS_DATUM"] = pd.to_datetime(df["MESS_DATUM"])
        drop_cols = [c for c in df.columns if c.strip() in ("STATIONS_ID", "QN")]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)

    # Merge on timestamp
    merged = (
        df_temp.set_index("MESS_DATUM")
        .join(df_prec.set_index("MESS_DATUM"), how="inner")
        .join(df_wind.set_index("MESS_DATUM"), how="inner")
    )
    if merged.empty:
        raise RuntimeError(
            f"[weather_generator] Merged weather data is empty for directory: {data_dir}"
        )

    print(f"[weather_generator] Merged weather data has {len(merged)} rows")

    # Define simulation window (time of day; we apply it per date)
    start_time = datetime(2019, 6, 1, 8, 20)
    end_time = datetime(2019, 6, 1, 18, 20)
    delta = timedelta(minutes=10)

    records: List[dict] = []

    # Group by calendar date and filter within the daily time window
    grouped = merged.groupby(merged.index.date)
    for date, group in grouped:
        day_data = group.sort_index()
        mask = (day_data.index.time >= start_time.time()) & (
            day_data.index.time <= end_time.time()
        )
        day_subset = day_data.loc[mask]
        for ts, row in day_subset.iterrows():
            records.append(
                {
                    "timestamp": ts.isoformat() + "Z",
                    "data": row.to_dict(),
                }
            )

    if not records:
        raise RuntimeError(
            f"[weather_generator] No weather records within simulation window "
            f"for directory: {data_dir}"
        )

    print(
        f"[weather_generator] Loaded {len(records)} weather records; "
        f"first timestamp = {records[0]['timestamp']}"
    )
    return records


def build_weather_dataset() -> List[dict]:
    """Construct the list of weather observations for streaming.
    """
    path = os.environ.get("WEATHER_ROOT")
    if not path:
        raise RuntimeError(
            "[weather_generator] WEATHER_DATA_PATH not set. "
            "Set it to the cleaned weather directory, e.g. "
            "WEATHER_DATA_PATH=./hugging_face/weather_berlin-tempel/cleaned"
        )

    print(f"[weather_generator] Using WEATHER_DATA_PATH={path}")
    records = _load_weather_records(path)
    return records


app = FastAPI(title="Weather Generator Service")

WEATHER_DATA: List[dict] = []  # global dataset


@app.on_event("startup")
async def startup_event() -> None:
    global WEATHER_DATA
    WEATHER_DATA = build_weather_dataset()
    print(
        f"[weather_generator] Startup complete. Weather dataset contains "
        f"{len(WEATHER_DATA)} records; first timestamp = {WEATHER_DATA[0]['timestamp']}"
    )


@app.websocket("/weather_stream")
async def weather_stream(websocket: WebSocket) -> None:
    """WebSocket endpoint that streams raw weather observations.
    """
    await websocket.accept()
    print("[weather_generator] Client connected to /weather_stream")
    paused: bool = False
    running: bool = False
    speed: float = 2.0  # seconds per 10-minute step
    index: int = 0
    dataset = WEATHER_DATA

    async def send_records() -> None:
        nonlocal index, running, paused, speed, dataset
        while True:
            if not running:
                await asyncio.sleep(0.1)
                continue
            if paused:
                await asyncio.sleep(0.1)
                continue
            if index >= len(dataset):
                await websocket.send_json({"type": "end_of_data"})
                print("[weather_generator] Sent end_of_data; stopping run")
                running = False
                await asyncio.sleep(0.1)
                continue
            entry = dataset[index]
            payload = {
                "type": "weather",
                "timestamp": entry.get("timestamp"),
                "data": entry.get("data"),
            }
            await websocket.send_json(payload)
            print(
                f"[weather_generator] Sent weather record index={index}, "
                f"timestamp={payload['timestamp']}"
            )
            index += 1
            await asyncio.sleep(float(speed))

    async def receive_controls() -> None:
        nonlocal running, paused, speed, index
        while True:
            try:
                msg = await websocket.receive_json()
            except WebSocketDisconnect:
                running = False
                print("[weather_generator] Client disconnected from /weather_stream")
                break
            if not isinstance(msg, dict):
                continue
            print(f"[weather_generator] Received control message: {msg}")
            mtype = msg.get("type")
            if mtype == "start":
                index = 0
                running = True
                paused = False
                speed = float(msg.get("speed", speed))
                print(
                    f"[weather_generator] Start command: speed={speed}, resetting index"
                )
            elif mtype == "pause":
                paused = True
                print("[weather_generator] Pause command received")
            elif mtype == "resume":
                paused = False
                print("[weather_generator] Resume command received")
            elif mtype == "set_speed":
                new_speed = msg.get("speed")
                if new_speed is not None:
                    speed = float(new_speed)
                    print(
                        f"[weather_generator] Set_speed command: new speed={speed}"
                    )
            elif mtype == "stop":
                running = False
                await websocket.send_json({"type": "end_of_data"})
                print("[weather_generator] Stop command received; sent end_of_data")
            else:
                print(f"[weather_generator] Unknown control type: {mtype}")

    sender = asyncio.create_task(send_records())
    receiver = asyncio.create_task(receive_controls())
    done, pending = await asyncio.wait(
        [sender, receiver], return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()
