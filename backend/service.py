"""
Backend service: consumes traffic + weather streams, runs models and
streams predictions + loss history.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import websockets
import torch

from backend.inference import InferencePipeline
from backend.models import load_raw_model, load_enriched_model
from shared.utils.losses import MaskedMAEFocalLoss
from shared.utils.lbcs import load_lbcs_onehot
from shared.utils.weather_encoder import WeatherEncoder
from shared.utils.mask import load_mask

from .utils import prepare_inputs_raw, prepare_inputs_enriched  # noqa: F401

from dotenv import load_dotenv

load_dotenv()


def log(msg: str) -> None:
    print(msg)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_ROOT = os.environ.get("DATA_ROOT")
WEATHER_ROOT = os.environ.get(
    "WEATHER_ROOT")
LBCS_PATH = os.environ.get(
    "LBCS_PATH")
MODELS_DIR = os.environ.get(
    "MODEL_PARAMETERS_DIR")


HISTORY_STEPS = 12
FUTURE_STEPS = 4
LBCS_CHANNELS = 9

TRAFFIC_GENERATOR_URL = os.environ.get(
    "TRAFFIC_GENERATOR_URL"
)
WEATHER_GENERATOR_URL = os.environ.get(
    "WEATHER_GENERATOR_URL"
)


def build_lbcs_hist(H: int, W: int) -> np.ndarray:
    """Preload LBCS one-hot and repeat over history.

    Uses the same upsampling logic as training. Fails if the channel
    count does not match LBCS_CHANNELS to avoid silently hiding errors.
    """
    lbcs_tensor = load_lbcs_onehot(LBCS_PATH, H, W)  # (C, H, W)
    lbcs_np = lbcs_tensor.cpu().numpy()

    if lbcs_np.shape[0] != LBCS_CHANNELS:
        raise ValueError(
            f"LBCS tensor has {lbcs_np.shape[0]} channels, "
            f"expected {LBCS_CHANNELS}"
        )

    rep = np.repeat(lbcs_np[None, ...], HISTORY_STEPS, axis=0)
    return rep.reshape(HISTORY_STEPS * LBCS_CHANNELS, H, W)


def build_weather_vec_for_history_index(state: Any) -> np.ndarray:
    """
    Build the weather feature vector for the current history window.

    Uses WeatherEncoder.encode_timestamp(ts), exactly like training.
    Fails loudly if anything is missing or malformed.
    """
    if getattr(state, "weather_encoder", None) is None:
        raise RuntimeError("WeatherEncoder is not initialised")

    if not getattr(state, "traffic_timestamps", None):
        raise RuntimeError("No traffic timestamps available for weather_vec")

    idx = state.current_index
    if idx >= len(state.traffic_timestamps):
        raise IndexError(
            f"current_index {idx} out of range for traffic_timestamps "
            f"(len={len(state.traffic_timestamps)})"
        )

    ts_str = state.traffic_timestamps[idx]
    if not ts_str:
        raise RuntimeError(f"Empty traffic timestamp at index {idx}")

    try:
        ts = pd.to_datetime(ts_str.replace("Z", ""))
    except Exception as exc:
        raise ValueError(f"Cannot parse traffic timestamp {ts_str!r}") from exc

    vec = state.weather_encoder.encode_timestamp(ts)
    if vec is None:
        raise RuntimeError(f"WeatherEncoder returned None for timestamp {ts}")

    vec = np.asarray(vec, dtype=np.float32)
    if vec.ndim != 1:
        raise ValueError(
            f"weather_vec has wrong shape {vec.shape}, expected 1D vector"
        )
    return vec


def reset_run(state: Any) -> None:
    state.run_id += 1
    log(f"[reset_run] Starting new run: run_id={state.run_id}")
    state.loss_history.clear()
    state.current_index = 0
    state.running = False
    state.paused = False
    state.traffic_frames.clear()
    state.traffic_timestamps.clear()
    state.weather_records.clear()
    state.traffic_end = False
    state.weather_end = False


def append_loss(state: Any, frame_index: int, raw_loss: float, enr_loss: float) -> None:
    entry = {
        "run_id": state.run_id,
        "frame_index": frame_index,
        "raw": float(raw_loss),
        "enriched": float(enr_loss),
    }
    state.loss_history.append(entry)
    log(
        f"[append_loss] frame={frame_index} raw={entry['raw']:.4f} "
        f"enriched={entry['enriched']:.4f}"
    )


app = FastAPI(title="Backend Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[startup] Using device: {device}")

    weather_encoder = WeatherEncoder(WEATHER_ROOT)
    log(f"[startup] WeatherEncoder vars: {weather_encoder.vars}")

    mask_path = os.path.join(DATA_ROOT, "BERLIN_reduced")
    mask_tensor = load_mask(mask_path, device)  # (1,1,H,W)
    app.state.mask_tensor = mask_tensor
    H, W = mask_tensor.shape[-2:]
    log(f"[startup] Mask shape: {mask_tensor.shape}")

    app.state.lbcs_hist = build_lbcs_hist(H, W)
    log(f"[startup] LBCS history shape: {app.state.lbcs_hist.shape}")

    app.state.mask_np = mask_tensor.squeeze().detach().cpu().numpy().astype(bool)
    app.state.weather_encoder = weather_encoder
    app.state.device = device

    app.state.raw_model = load_raw_model(device, parameters_dir=MODELS_DIR)
    app.state.enriched_model = load_enriched_model(
        device, weather_dim=len(weather_encoder.vars), parameters_dir=MODELS_DIR
    )
    log("[startup] Models loaded")

    app.state.test_samples = []
    app.state.run_id = 0
    app.state.loss_history = []
    app.state.current_index = 0
    app.state.running = False
    app.state.paused = False
    app.state.speed = 1.0
    app.state.traffic_frames = []
    app.state.traffic_timestamps = []
    app.state.weather_records = []
    app.state.traffic_end = False
    app.state.weather_end = False
    app.state.traffic_task = None
    app.state.weather_task = None
    app.state.traffic_ws = None
    app.state.weather_ws = None

    app.state.inference_pipeline = InferencePipeline(
        app.state.raw_model,
        app.state.enriched_model,
        device,
    )
    log("[startup] InferencePipeline ready")

    asyncio.create_task(connect_to_generators(app.state))


async def connect_to_generators(state: Any) -> None:
    try:
        log(f"[connect] traffic → {TRAFFIC_GENERATOR_URL}")
        state.traffic_ws = await websockets.connect(TRAFFIC_GENERATOR_URL, max_size=None)
        log("[connect] Traffic connected")
    except Exception as e:
        log(f"[connect] Traffic connection failed: {e}")
        state.traffic_ws = None

    try:
        log(f"[connect] weather → {WEATHER_GENERATOR_URL}")
        state.weather_ws = await websockets.connect(WEATHER_GENERATOR_URL, max_size=None)
        log("[connect] Weather connected")
    except Exception as e:
        log(f"[connect] Weather connection failed: {e}")
        state.weather_ws = None

    if state.traffic_ws is not None:
        state.traffic_task = asyncio.create_task(traffic_receiver(state))
    if state.weather_ws is not None:
        state.weather_task = asyncio.create_task(weather_receiver(state))


async def traffic_receiver(state: Any) -> None:
    ws = state.traffic_ws
    if ws is None:
        return
    log("[traffic_receiver] started")
    while True:
        try:
            msg = await ws.recv()
        except Exception as e:
            log(f"[traffic_receiver] recv exception: {e!r}")
            state.traffic_end = True
            break
        try:
            data = json.loads(msg)
        except Exception as e:
            log(f"[traffic_receiver] bad JSON: {e!r}")
            continue
        mtype = data.get("type")
        if mtype == "frame":
            frame_list = data.get("frame")
            if frame_list is None:
                continue
            arr = np.asarray(frame_list, dtype=np.float32)
            state.traffic_frames.append(arr)
            state.traffic_timestamps.append(data.get("timestamp", ""))
            idx = len(state.traffic_frames) - 1
            log(
                f"[traffic_receiver] frame {idx} ts={data.get('timestamp','')} "
                f"shape={arr.shape}"
            )
        elif mtype == "end_of_data":
            log("[traffic_receiver] end_of_data")
            state.traffic_end = True
            continue


async def weather_receiver(state: Any) -> None:
    ws = state.weather_ws
    if ws is None:
        return
    log("[weather_receiver] started")
    while True:
        try:
            msg = await ws.recv()
        except Exception as e:
            log(f"[weather_receiver] recv exception: {e!r}")
            state.weather_end = True
            break
        try:
            data = json.loads(msg)
        except Exception:
            continue
        mtype = data.get("type")
        if mtype == "weather":
            record = {"timestamp": data.get(
                "timestamp"), "data": data.get("data", {})}
            state.weather_records.append(record)
            log(f"[weather_receiver] ts={record['timestamp']}")
        elif mtype == "end_of_data":
            log("[weather_receiver] end_of_data")
            state.weather_end = True
            break


async def send_generator_command(state: Any, generator: str, message: Dict[str, Any]) -> None:
    ws = state.traffic_ws if generator == "traffic" else state.weather_ws
    if ws is None:
        log(f"[send_command] no WS for {generator}")
        return
    log(f"[send_command] {generator}: {message}")
    try:
        await ws.send(json.dumps(message))
    except Exception as e:
        log(f"[send_command] failed → {generator}: {e}")


@app.websocket("/stream")
async def prediction_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    log("[stream] client connected")
    state = app.state

    async def handle_controls() -> None:
        while True:
            try:
                msg = await websocket.receive_json()
            except WebSocketDisconnect:
                state.running = False
                log("[stream] client disconnected")
                break
            if not isinstance(msg, dict):
                continue
            log(f"[controls] {msg}")
            mtype = msg.get("type")
            if mtype == "start":
                reset_run(state)
                state.speed = float(msg.get("speed", 1.0))
                state.running = True
                state.paused = False
                log(f"[controls] start speed={state.speed} run_id={state.run_id}")
                await send_generator_command(
                    state, "traffic", {"type": "start", "speed": state.speed}
                )
                await send_generator_command(
                    state, "weather", {"type": "start",
                                       "speed": state.speed * 2.0}
                )
            elif mtype == "pause":
                state.paused = True
                log("[controls] pause")
                await send_generator_command(state, "traffic", {"type": "pause"})
                await send_generator_command(state, "weather", {"type": "pause"})
            elif mtype == "resume":
                state.paused = False
                log("[controls] resume")
                await send_generator_command(state, "traffic", {"type": "resume"})
                await send_generator_command(state, "weather", {"type": "resume"})
            elif mtype == "set_speed":
                try:
                    val = float(msg.get("value"))
                except (TypeError, ValueError):
                    log("[controls] bad speed")
                    continue
                state.speed = val
                log(f"[controls] set_speed={state.speed}")
                await send_generator_command(
                    state, "traffic", {"type": "set_speed", "speed": val}
                )
                await send_generator_command(
                    state, "weather", {"type": "set_speed", "speed": val * 2.0}
                )
            elif mtype == "stop":
                state.running = False
                state.paused = False
                log("[controls] stop")
                await send_generator_command(state, "traffic", {"type": "stop"})
                await send_generator_command(state, "weather", {"type": "stop"})

    def _flatten_history_for_enriched(x_hist: np.ndarray) -> np.ndarray:
        T, Hx, Wx, Cx = x_hist.shape
        return x_hist.transpose(0, 3, 1, 2).reshape(T * Cx, Hx, Wx)

    async def produce_predictions() -> None:
        while True:
            if not state.running:
                await asyncio.sleep(0.1)
                continue
            if state.paused:
                await asyncio.sleep(0.1)
                continue

            required_frames = state.current_index + HISTORY_STEPS + FUTURE_STEPS
            if required_frames > len(state.traffic_frames):
                await asyncio.sleep(0.05)
                continue

            hist_frames = state.traffic_frames[
                state.current_index: state.current_index + HISTORY_STEPS
            ]
            log(
                f"[predict] index={state.current_index} "
                f"hist_frames={len(hist_frames)}"
            )

            H, W, C = hist_frames[0].shape
            x_hist = np.stack(hist_frames, axis=0)  # (T,H,W,C)
            log(f"[predict] x_hist shape={x_hist.shape}")

            traffic_flat = _flatten_history_for_enriched(x_hist)  # (T*C,H,W)
            enr_concat = traffic_flat
            if state.lbcs_hist is not None:
                enr_concat = np.concatenate(
                    (traffic_flat, state.lbcs_hist), axis=0)
            log(f"[predict] enr_concat shape={enr_concat.shape}")

            # Strict weather vector based on history timestamp (no zero fallbacks)
            weather_vec = build_weather_vec_for_history_index(state)
            log(f"[predict] weather_vec len={len(weather_vec)}")

            log(f"[predict] weather_vec len={len(weather_vec)}")

            sample = {
                "x_raw": x_hist,
                "x_enriched": enr_concat,
                "weather_vec": weather_vec,
            }

            preds = state.inference_pipeline.process_sample(sample)
            raw_pred = preds.get("raw")
            enr_pred = preds.get("enriched")
            log(
                f"[predict] raw_pred shape={raw_pred.shape}, "
                f"enriched_pred={None if enr_pred is None else enr_pred.shape}"
            )

            gt_frames = state.traffic_frames[
                state.current_index + HISTORY_STEPS:
                state.current_index + HISTORY_STEPS + FUTURE_STEPS
            ]
            gt = np.stack([f[:, :, 0] for f in gt_frames], axis=0)  # (F,H,W)

            criterion = MaskedMAEFocalLoss(state.mask_tensor.to(state.device))
            with torch.no_grad():
                raw_pred_tensor = torch.tensor(
                    raw_pred, dtype=torch.float32, device=state.device
                ).unsqueeze(0)
                gt_tensor = torch.tensor(
                    gt, dtype=torch.float32, device=state.device
                ).unsqueeze(0)

                raw_loss = criterion(raw_pred_tensor, gt_tensor)
                raw_loss_val = float(raw_loss.item())

                enr_pred_tensor = torch.tensor(
                    enr_pred, dtype=torch.float32, device=state.device
                ).unsqueeze(0)
                enr_loss = criterion(enr_pred_tensor, gt_tensor)
                enr_loss_val = float(enr_loss.item())

            log(
                f"[predict] loss raw={raw_loss_val:.4f} "
                f"enriched={enr_loss_val:.4f}"
            )

            def compute_classes(pred: np.ndarray,
                    truth: np.ndarray,
                    mask: np.ndarray,
                    eps_zero: float = 1e-3,
                    eps_val: float = 0.05) -> np.ndarray:
                """
                Per-cell classification:
                0 = outside mask
                1 = both ~zero
                2 = both > 0 and with small error
                3 = rest (incorrect)
                """
                classes = np.zeros_like(mask, dtype=np.int32)
                valid = mask > 0

                # “Zero” ~ very small value
                pred_zero = (np.abs(pred) < eps_zero)
                truth_zero = (np.abs(truth) < eps_zero)

                correct_zero = truth_zero & pred_zero & valid

                # Both with traffic and small error
                both_nonzero = (~truth_zero) & (~pred_zero) & valid
                abs_err = np.abs(pred - truth)
                correct_nonzero = both_nonzero & (abs_err < eps_val)

                incorrect = valid & ~(correct_zero | correct_nonzero)

                classes[correct_zero] = 1
                classes[correct_nonzero] = 2
                classes[incorrect] = 3

                return classes



            raw_classes = compute_classes(raw_pred[0], gt[0], state.mask_np)
            enr_classes = compute_classes(enr_pred[0], gt[0], state.mask_np)



            append_loss(state, state.current_index, raw_loss_val, enr_loss_val)

            if not state.traffic_timestamps:
                raise RuntimeError("No traffic timestamps available when building frame message")

            if state.current_index + HISTORY_STEPS >= len(state.traffic_timestamps):
                raise IndexError(
                    f"Timestamp index {state.current_index + HISTORY_STEPS} "
                    f"out of range for traffic_timestamps of length {len(state.traffic_timestamps)}"
                )

            timestamp = state.traffic_timestamps[state.current_index + HISTORY_STEPS]

            msg = {
                "type": "frame",
                "frame_index": state.current_index,
                "timestamp": timestamp,
                "raw_classes": raw_classes.tolist(),
                "enriched_classes": enr_classes.tolist(),
                "loss": {
                    "metric": "masked_mae_focal",
                    "raw": raw_loss_val,
                    "enriched": enr_loss_val,
                },
            }
            try:
                await websocket.send_json(msg)
                log(f"[predict] sent frame {state.current_index} ts={timestamp}")
            except Exception:
                log("[predict] send failed, stopping")
                state.running = False
                break

            state.current_index += 1
            await asyncio.sleep(state.speed)

    control_task = asyncio.create_task(handle_controls())
    prediction_task = asyncio.create_task(produce_predictions())
    done, pending = await asyncio.wait(
        [control_task, prediction_task], return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()


@app.get("/metrics/loss_history")
async def get_loss_history() -> JSONResponse:
    state = app.state
    current = state.run_id
    history = [e for e in state.loss_history if e["run_id"] == current]
    log(f"[loss_history] {len(history)} entries for run {current}")
    return JSONResponse({"metric": "masked_mae_focal", "history": history})
