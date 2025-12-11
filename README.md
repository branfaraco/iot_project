# IoT Traffic Forecasting Demo – Technical Documentation

This repository implements an IoT-style system for short-term traffic
forecasting on the Berlin Traffic4Cast grid. The system:

- Streams pre-recorded **traffic** and **weather** data as if they were
  live sensor feeds.
- Runs two trained models in a **backend** service:
  - a **baseline** traffic-only model,
  - an **enriched** model with traffic + LBCS land-use + weather.
- Exposes predictions and masked losses to a **web frontend** for
  interactive exploration.

The main paper gives a high-level description of the system. This
document contains the complete technical specification of the app
(endpoints, JSON formats, environment variables and runtime behaviour).

---

## 1. Components and data flow

High-level architecture (same as in the paper):

```text
        +--------------------+
        |   Web frontend     |
        |  (React/TypeScript)|
        +----------+---------+
                   |   WebSocket: /stream
                   |   HTTP: GET /metrics/loss_history
                   v
        +----------+---------+
        |    Backend API     |
        |  (FastAPI service) |
        +----------+---------+
           ^              ^
           | WS           | WS
   /traffic_stream  /weather_stream
           |              |
+----------+--+      +----+-----------+
| Traffic gen |      | Weather gen    |
|  (FastAPI)  |      |   (FastAPI)    |
+------+------+      +--------+-------+
       |                        |
       |                        |
 HDF5 test files         Cleaned weather
 from BERLIN_reduced     CSVs (Tempelhof)
 (Hugging Face)          (Hugging Face)
````

At a high level:

* **Traffic generator** reads test HDF5 files from `BERLIN_reduced/test`
  and streams raw 5-minute frames.
* **Weather generator** reads cleaned CSVs from
  `weather_berlin_tempel/cleaned` and streams 10-minute weather records.
* **Backend**:

  * receives both streams,
  * applies the same preprocessing used in training,
  * runs the baseline and enriched models,
  * sends classification maps and per-frame losses to the frontend via
    `/stream`,
  * exposes a metrics snapshot via `GET /metrics/loss_history`.
* **Frontend** (React) controls playback and visualises predictions and
  losses.

---

## 2. Environment and configuration

The backend and generators rely on a `.env` file.

Key assumptions:

* Berlin test HDF5 files (temporal-cropped to slots 100–220) live in
  `./hugging_face/BERLIN_reduced/test`.
* Training/validation reduced files are in
  `./hugging_face/BERLIN_reduced/data`.
* Cleaned weather data (Tempelhof station) live in
  `./hugging_face/weather_berlin-tempel/cleaned` with a `test/` subdir.

For downloading this data use the `data_generation/download_from_hf.py` file with the token of the report.

## 3. Data preprocessing (training vs streaming)

### 3.1 Traffic preprocessing

The training-time preprocessing for Berlin is implemented in
`shared/utils/data_reduction.py` and consists of:

1. **Temporal crop** (all splits):

   * From 288 five-minute slots (00:00–24:00) to slots **100–220**
     inclusive, corresponding to **08:20–18:20**.
   * Implemented as:

     ```python
     T_START, T_END = 100, 220
     arr_proc = temporal_crop(arr, T_START, T_END)
     # temporal_crop(arr, t0, t1) = arr[t0:t1]
     ```

2. **Spatial crop** (for reduced training/val data):

   * Original grid: (H, W) = (495, 436).
   * Sub-grid indices (multiples of 8):

     ```python
     r0, r1 = 10, 274
     c0, c1 = 0, 432
     # (H_roi, W_roi) = (r1-r0, c1-c0) = (264, 432)
     ```

3. **Normalisation**:

   * All 8 channels scaled from `[0, 255]` to `[0, 1]`:

     ```python
     def normalize(arr):
         return arr.astype(np.float32) / 255.0
     ```

Training uses the **spatially cropped + normalised** tensors from
`BERLIN_reduced/data`.

Streaming behaves as follows:

* **Traffic generator** streams the **temporally cropped, full-grid**
  frames from `BERLIN_reduced/test`, shape `(T', 495, 436, 8)`.
* **Backend** re-applies:

  * the same **spatial crop** `(10:274, 0:432)` to get `(264, 432, 8)`,
  * the same **normalisation** by `/255.0`.

This ensures the streaming inputs match the trained model expectations.

### 3.2 Land-use (LBCS) preprocessing

* `LBCS_PATH` points to a JSON file that stores a one-hot LBCS tensor
  for Berlin at the same `(H, W)` as the traffic ROI `(264, 432)`.

* On backend startup:

  ```python
  lbcs_tensor = load_lbcs_onehot(LBCS_PATH, H, W)  # (C, H, W)
  lbcs_np = lbcs_tensor.cpu().numpy()              # float32
  ```

* It is then padded or trimmed to `LBCS_CHANNELS` (default 9) and
  repeated across `HISTORY_STEPS` to obtain:

  ```python
  lbcs_hist.shape == (HISTORY_STEPS * LBCS_CHANNELS, H, W)
  ```

* At inference time, this `lbcs_hist` is concatenated with the flattened
  traffic history for the enriched model.

### 3.3 Weather preprocessing

Weather preprocessing is centralised in `shared/utils/weather_encoder.py`:

* It defines a `WeatherEncoder` class that:

  * knows which variables to use (e.g. pressure, temperature, wind, etc.),
  * loads cleaned CSVs from `WEATHER_ROOT`,
  * constructs a fixed-length numeric vector for each timestamp.

* The **same encoder class** is used:

  * When training the enriched model (offline).
  * In the backend during streaming, to ensure the weather vector has
    the same dimension and semantics.

Streaming path:

* **Weather generator**:

  * Loads cleaned CSVs from `${WEATHER_DATA_PATH}/test`:

    * `temp.csv`, `precip.csv`, `wind.csv` (merged on `MESS_DATUM`).
  * Emits raw numeric dicts:

    ```json
    {
      "type": "weather",
      "timestamp": "2019-06-03T09:20:00Z",
      "data": {
        "P0": 1007.8,
        "TT": 26.1,
        "TM": 32.3,
        "TX": 44.8,
        "FF": 13.2,
        "RR": 1.1,
        "DD": 60.0
        // etc.
      }
    }
    ```

* **Backend**:

  * Takes the last record in `state.weather_records`.

  * Builds a vector in the same order as `weather_encoder.vars`:

    ```python
    raw = state.weather_records[-1]["data"]
    vec = [float(raw.get(var, 0.0)) for var in state.weather_encoder.vars]
    weather_vec = np.array(vec, dtype=np.float32)  # shape (weather_dim,)
    ```

  * This `weather_vec` is passed to the FiLM-conditioned enriched model.

---

## 4. Backend API

### 4.1 WebSocket: `/stream`

The main WebSocket endpoint for real-time predictions and control is
`/stream` on the backend service (`uvicorn backend.service:app`).

#### 4.1.1 Control messages (frontend → backend)

All control messages are JSON objects with a `type` field:

* **Start a run**

  ```json
  { "type": "start", "speed": 0.5 }
  ```

  * `speed` is the real-time scaling in seconds per 5-minute traffic
    step.
  * Backend:

    * calls `reset_run(state)`,
    * sets `state.speed = 0.5`,
    * sets `state.running = True`, `state.paused = False`,
    * sends corresponding `start` commands to both generators (see below).

* **Pause**

  ```json
  { "type": "pause" }
  ```

  * Backend sets `state.paused = True` and forwards `{"type": "pause"}` to
    both generators.

* **Resume**

  ```json
  { "type": "resume" }
  ```

  * Backend sets `state.paused = False` and forwards `{"type": "resume"}`.

* **Change speed**

  ```json
  { "type": "set_speed", "value": 1.0 }
  ```

  * Backend sets `state.speed = 1.0` and sends:

    * `{"type": "set_speed", "speed": 1.0}` to traffic generator,
    * `{"type": "set_speed", "speed": 2.0}` to weather generator (2×).

* **Stop**

  ```json
  { "type": "stop" }
  ```

  * Backend sets `state.running = False`, `state.paused = False` and
    sends `{"type": "stop"}` to both generators.

Any unknown `type` is logged and ignored.

#### 4.1.2 Prediction messages (backend → frontend)

For each history window, once models and losses are computed, the
backend sends:

```json
{
  "type": "frame",
  "frame_index": 64,
  "timestamp": "2019-06-01T09:40:00Z",
  "raw_classes": [[0, 1, 2, ...], [...]],
  "enriched_classes": [[0, 1, 3, ...], [...]],
  "loss": {
    "metric": "masked_mae_focal",
    "raw": 0.0012,
    "enriched": 104260872276279296.0
  }
}
```

* `frame_index`: index of the history window within the current run.

* `timestamp`: timestamp of the **first future frame** used for
  evaluation.

* `raw_classes`, `enriched_classes`:

  * 2D integer maps of shape `(H, W)` (here `(264, 432)`).
  * Encoded as:

    * `0` – non-road cells (mask = 0),
    * `1` – road cells where ground truth and prediction are both zero,
    * `2` – road cells where ground truth and prediction are both > 0,
    * `3` – any other valid road cell (mismatch).

* `loss`:

  * `metric`: `"masked_mae_focal"`.
  * `raw`: scalar loss for baseline model.
  * `enriched`: scalar loss for enriched model.

The frontend uses these frames to colour the grid and to update the
metric chart by polling the HTTP endpoint below.

### 4.2 HTTP: `GET /metrics/loss_history`

Returns all per-frame losses for the **current** `run_id`:

```http
GET /metrics/loss_history
```

Response:

```json
{
  "metric": "masked_mae_focal",
  "history": [
    {
      "run_id": 1,
      "frame_index": 0,
      "raw": 0.0015,
      "enriched": 0.0014
    },
    {
      "run_id": 1,
      "frame_index": 1,
      "raw": 0.0013,
      "enriched": 0.0012
    }
    // ...
  ]
}
```

* `run_id` increments each time `reset_run(state)` is called (on
  `"start"`).
* The frontend periodically calls this endpoint (e.g. every second) to
  update the loss history plot.



## 5. Traffic generator API

The traffic generator is a small FastAPI/WebSocket service (see
`backend/traffic_generator.py`).

### 5.1 Startup

On startup:

1. Reads `TRAFFIC_DATA_PATH` from the environment.

2. Scans the directory for `*.h5` files (e.g.
   `2019-06-24_BERLIN_8ch.h5`, …).

3. For each file:

   * Assumes a dataset `"array"` with shape `(T, 495, 436, 8)`.
   * Assumes it has already been temporally cropped to slots `100–220`.
   * Wraps each frame as a dict:

     ```python
     {
       "frame": arr[i],       # (495, 436, 8)
       "timestamp": ISO8601,
       "file": fname
     }
     ```

4. Concatenates across all test files into a single list `DATASET`.

### 5.2 WebSocket: `/traffic_stream`

Endpoint: `/traffic_stream`

#### 5.2.1 Control messages (backend → generator)

Same control protocol as backend `/stream`, but here the backend is the
client.

* Start:

  ```json
  { "type": "start", "speed": 0.5 }
  ```

  * Resets internal `index = 0`, `running = True`, `paused = False`.
  * Sets `speed` = seconds per 5-minute traffic step.

* Pause:

  ```json
  { "type": "pause" }
  ```

* Resume:

  ```json
  { "type": "resume" }
  ```

* Change speed:

  ```json
  { "type": "set_speed", "speed": 1.0 }
  ```

* Stop:

  ```json
  { "type": "stop" }
  ```

  * Stops emitting and sends a final `{"type": "end_of_data"}`.

#### 5.2.2 Data messages (generator → backend)

For each frame:

```json
{
  "type": "frame",
  "file": "2019-06-24_BERLIN_8ch.h5",
  "index": 123,
  "timestamp": "2019-06-01T09:40:00Z",
  "frame": [[[...], [...], ...]]   // shape (495, 436, 8)
}
```

At the end of the dataset:

```json
{ "type": "end_of_data" }
```

The backend stores the raw `frame` arrays in `state.traffic_frames` and
timestamps in `state.traffic_timestamps`, then applies its own spatial
crop and normalisation before feeding the models.

---

## 6. Weather generator API

The weather generator is another FastAPI/WebSocket service with the same structure.

### 6.1 Startup

On startup:

1. Reads `WEATHER_DATA_PATH` from environment.
2. Looks for `${WEATHER_DATA_PATH}/test/{temp.csv, precip.csv, wind.csv}`.

   * parse `MESS_DATUM` as timestamps,
   * drop metadata columns (`STATIONS_ID`, `QN`, …),
   * merge on `MESS_DATUM`,
   * filter times `08:20–18:20` for each available test day.
4. Produces a list of records:

   ```python
   {
     "timestamp": ts.isoformat() + "Z",
     "data": row.to_dict(),   # keys are weather variable names
   }
   ```

### 6.2 WebSocket: `/weather_stream`

#### 6.2.1 Control messages (backend → generator)

Same protocol as traffic:

```json
{ "type": "start", "speed": 1.0 }
{ "type": "pause" }
{ "type": "resume" }
{ "type": "set_speed", "speed": 2.0 }
{ "type": "stop" }
```

* Here, `speed` is seconds per **10-minute** weather step.
* Backend usually sends `speed * 2` compared to traffic so that the
  5-minute : 10-minute ratio is maintained.

#### 6.2.2 Data messages (generator → backend)

```json
{
  "type": "weather",
  "timestamp": "2019-06-03T09:20:00Z",
  "data": {
    "P0": 1007.8,
    "TT": 26.1,
    "TM": 32.3,
    "TX": 44.8,
    "FF": 13.2,
    "RR": 1.1,
    "DD": 60.0
  }
}
```

At the end:

```json
{ "type": "end_of_data" }
```

The backend appends these dicts to `state.weather_records` and uses only
the **latest** record to build the `weather_vec` for each prediction
step.

---

## 7. Backend internal state (summary)

At runtime, the backend uses the following key state fields
(`app.state`):

* `device`: `torch.device("cuda")` or `"cpu"`.
* `raw_model`: baseline U-Net loaded from `RAW_MODEL_WEIGHTS`.
* `enriched_model`: FiLM-conditioned U-Net loaded from
  `ENRICHED_MODEL_WEIGHTS`.
* `inference_pipeline`: wraps both models and runs inference on prepared
  samples.
* `weather_encoder`: `WeatherEncoder` instance for the streaming period.
* `mask_tensor`: road mask tensor `(1, 1, H, W)` on `device`.
* `mask_np`: road mask as boolean array `(H, W)`.
* `lbcs_hist`: flattened LBCS history
  `(HISTORY_STEPS * LBCS_CHANNELS, H, W)`.
* `traffic_frames`: list of raw frames `(495, 436, 8)` from generator.
* `traffic_timestamps`: list of ISO timestamps for traffic frames.
* `weather_records`: list of `{timestamp, data}` dicts from generator.
* `traffic_ws`, `weather_ws`: active WebSocket client connections to the
  generators.
* `traffic_task`, `weather_task`: background receiver tasks.
* `running`, `paused`, `speed`, `current_index`: main loop control.
* `run_id`: integer run identifier, increments on `"start"`.
* `loss_history`: list of loss entries:

  ```python
  {
    "run_id": int,
    "frame_index": int,
    "raw": float,
    "enriched": float,
  }
  ```

These fields are read or updated by the `/stream` handler and the
`/metrics/loss_history` endpoint.

---

## 8. Running the system locally

Assuming you have Python 3.11 and Node.js:

### 8.1 Backend + generators

```bash
# create and activate virtual env
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# run traffic generator
uvicorn backend.traffic_generator:app --host 0.0.0.0 --port 8001

# run weather generator (separate terminal)
uvicorn backend.weather_generator:app --host 0.0.0.0 --port 8002

# run backend service (separate terminal)
uvicorn backend.service:app --host 0.0.0.0 --port 8000
```

Ensure `.env` is in the repository root (or wherever `load_dotenv()`
expects it) and paths are correct.

### 8.2 Frontend

From the frontend directory (e.g. `frontend/`):

```bash
npm install
npm run dev  
```

By default, the frontend is assumed to run on `http://localhost:8080`
and to connect to the backend at:

* `ws://localhost:8000/stream`
* `http://localhost:8000/metrics/loss_history`

I

