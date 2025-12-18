# Backend service

This folder implements the backend of the Traffic Prediction Visualiser.
The backend is a Python FastAPI application that receives streaming
traffic and weather data, applies preprocessing and runs two
pre‑trained U‑Net models (baseline and enriched) in real time.  It
exposes a WebSocket endpoint for predictions and an HTTP endpoint for
per‑frame loss metrics. 

## Architecture

At runtime the backend connects to the two generator services: the traffic
generator that sends 5‑minute frames of the test split and the weather
generator that sends 10‑minute meteorological records.  When
both streams have provided enough data to build a history window
(currently H = 12 past frames, or one hour) the backend performs the
following steps to behave as in training:

1. **Spatial and temporal preprocessing** – Incoming traffic frames
   are cropped to a fixed 264×432 subgrid , scaled from
   `[0, 255]` to `[0, 1]` and stacked along the channel axis.
   Land‑use channels (nine one‑hot layers derived from the LBCS
   grid) are repeated across the history dimension and concatenated
   with the traffic history.  A weather vector is
   computed from the timestamp of the most recent weather record via
   the `WeatherEncoder` from the `shared/utils/weather_encoder.py`
   module.
2. **Model execution** – Both the baseline U‑Net and the enriched
   U‑Net are applied to the preprocessed input.  The baseline model only consumes the traffic history, whereas the enriched model receives the concatenated
   traffic + land‑use tensor and is conditioned on the weather vector
   through FiLM layers.
3. **Loss computation** – Per‑frame errors are computed using the
   custom masked focal mean absolute error (MAE) loss.  These losses are logged in an in‑memory history and exposed via the `/metrics/loss_history` HTTP endpoint.
4. **Streaming to the frontend** – The predictions (aggregated
   volumes for the next 20 minutes) and loss values are sent to the
   web application over the `/stream` WebSocket.  

## Files and modules

- `service.py` – Entry point for the FastAPI application.  Defines
  the `/stream` WebSocket endpoint and a `/metrics/loss_history` HTTP
  endpoint, spawns tasks that consume the traffic and weather
  streams and coordinates model inference.
- `traffic_generator.py` – Standalone FastAPI service that reads
  preprocessed test frames from `DATA_ROOT/BERLIN_reduced/data/` and emits them over a  WebSocket.  It only sends the temporal window 08:20–18:20 (slots 100–220) to avoid long idle periods.
- `weather_generator.py` – Standalone FastAPI service that reads
  cleaned weather CSV files from `WEATHER_ROOT` and streams
  records at 10‑minute intervals. Precipitation, temperature
  and wind variables are used, encoded by the `WeatherEncoder`. It also only sends the temporal window 08:20–18:20.
- `inference/stream.py` – Core of the streaming logic.  It buffers
  incoming frames, synchronises timestamps between traffic and weather
  queues, builds input histories and calls the models.
- `inference/pipeline.py` – Helper functions for building the
  spatial input tensor and the weather embedding given a history
  window.
- `models/` – Simple wrappers around the pre‑trained UNet weights.
  `raw_model.py` loads the baseline model, while `enriched_model.py`
  loads the FiLM‑conditioned model.
- `utils/preprocess.py` – Contains the `preprocess_frame` function
  that crops and normalises a single traffic frame.  This helper is
  used by the backend and the generator scripts.

## API summary

The backend exposes two endpoints:

* **WebSocket `/stream`** – Clients connect here to start the
  simulation.  When the frontend sends a `start` command, the backend
  begins consuming frames and streams out JSON messages with the
  following fields:
  - `timestamp`: ISO‑8601 string of the last input frame;
  - `raw_pred` and `enriched_pred`: nested lists of shape
    `(F, H, W) = (4, 264, 432)` with predicted aggregated volumes;
  - `raw_loss` and `enriched_loss`: float errors for the current
    frame.
  Clients may also send `pause`, `resume` or `goto` commands to
  control playback.
* **HTTP `/metrics/loss_history`** – Returns a JSON object
  containing the recorded per‑frame losses for the current run.
