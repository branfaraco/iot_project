# Traffic Prediction Repository

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
document contains the technical specification of the app
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
```

At a high level:

* **Traffic generator** reads test HDF5 files from `BERLIN_reduced/test`
  and streams raw 5-minute frames.
* **Weather generator** reads cleaned CSVs from
  `weather_berlin_tempel/cleaned` and streams 10-minute weather records (not preprocessed, just filtering by days and dropping unusefull columns)

* **Backend**:

  * receives both streams,
  * applies the same preprocessing used in training,
  * runs the baseline and enriched models,
  * sends classification maps and per-frame losses to the frontend via
    `/stream`,
  * exposes a metrics snapshot via `GET /metrics/loss_history`.

* **Frontend** (React) controls playback and visualises predictions and
  losses.

* **Hugging face** hold all the data for the application working.
---

## 2. Organisation of documentation

This general README provides a high‑level overview and run
instructions.  Each top‑level folder contains a dedicated README
explaining its role in more detail. 

## 3. Repository structure

The repository is divided into several top‑level folders.  Each has its own
README explaining its contents and how it contributes to the system:

- `auxiliar/` – Helper notebooks and scripts used to analyse data,
  generate figures for the report and download the Hugging Face data.
- `backend/` – Python FastAPI application that consumes the traffic and
  weather streams, performs preprocessing and runs the baseline and
  enriched models.  It exposes WebSocket and HTTP endpoints for the
  frontend and logs per‑frame losses.  
- `data_generation/` – Scripts and notebooks to prepare the data.  The README contains all the explanation of the justifications of the data treatment.
- `hugging_face/` – Placeholder for the large datasets hosted on
  Hugging Face.  The contents must be downloaded via the script in
  `auxiliar/hugging_face_down/download_from_hf.py` with the token provided in the paper.
- `models/` – Pretrained weights for the baseline and enriched U‑Net
  models.  The backend loads these files at runtime.
- `shared/` – Library of shared code used by both the training scripts and
  the backend.  It contains the U‑Net model definitions, data
  preprocessing utilities, land‑use grid loading and weather encoding
  functions.  
- `training/` – Training scripts for reproducing the baseline and
  enriched models. 
- `web/` – React/TypeScript frontend that connects to the backend
  WebSocket, controls the simulated streams and visualises predictions
  and errors.

## Running the demo

The demo requires a Python environment and Node.js.  The following steps
assume you have cloned this repository, installed Python 3.11.1 (IMPORTANT FOR THE LIBRARIES)
and Node 18, and that you have downloaded the Hugging Face data into
`hugging_face/` (see below). Every .py file should be run from the root of the repository.

1. **Create and activate a virtual environment** 

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Python dependencies**.  

   ```bash
   pip install -r requirements.txt
   # install an appropriate PyTorch build separately, this is different for each computer deppending on the cuda instalation.
   ```

3. **Download the data from Hugging Face**. modify the token variable in download_from_hf.py and execute:
   ```bash
   
   python auxiliar/hugging_face_down/download_from_hf.py
   ```


4. **Configure environment variables**.  This should not cause problems since the .env is uploaded to the repo and the imports are relative. 
You should only change the `REPO_ROOT` variable.

5. **Launch the traffic and weather generators**.  In separate shells, run:

  ```bash
  uvicorn backend.traffic_generator:app --host 0.0.0.0 --port 8001
  uvicorn backend.weather_generator:app --host 0.0.0.0 --port 8002
  ```


6. **Launch the backend service**.  In another shell, run:

   ```bash
   uvicorn backend.service:app --reload --port 8000
   ```

7. **Start the web frontend**.  In another shell, run:

   ```bash
   cd web
   npm install
   npm run dev
   ```





