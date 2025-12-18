# Data generation

This directory contains scripts and notebooks used to define and reproduce the **reduced Berlin dataset** and the auxiliary artifacts used by the Traffic Prediction Visualiser. The project reduces the original Traffic4Cast data to a smaller spatial/temporal scope to keep the demo interpretable and to emphasize structural patterns where land-use context is expected to be informative.

The **authoritative preprocessing implementation** used by both training and inference is in `shared/utils/data_reduction.py`. This directory documents the rationale and provides standalone scripts/notebooks to reproduce the reduced assets.

## Traffic preprocessing (`traffic/`)

### Source data
The raw Traffic4Cast Berlin dataset consists of daily HDF5 files with:
- **288** five-minute frames per day (24 hours),
- a **495×436** spatial grid,
- **8 channels** stored as `uint8` in **[0, 255]** (image-like representation),
- a static mask (see below) indicating invalid / missing cells.

### Reductions and rationale
The notebooks under `traffic/` analyze the raw Berlin data and justify the following choices (implemented in `shared/utils/data_reduction.py`):

- **Temporal windowing**  
  Only frames from **08:20 to 18:20** are retained (indices **100–220**). Outside this window the signal is sparse and dominated by zeros, which makes qualitative inspection and metric interpretation less informative.

- **Spatial cropping**  
  The original 495×436 grid is cropped to a **264×432** subgrid centered on central Berlin. This removes low-activity borders and reduces computational cost while keeping the region with the most consistent activity.

- **Normalization**  
  Traffic channels are scaled from **[0, 255] → [0, 1]**, matching the convention used in most deep-learning baselines for Traffic4Cast-like data.

### Notebooks and scripts
- `original_traffic_data_analysis.ipynb`  
  Visualizes the raw dataset and supports the temporal/spatial reduction decisions (peak hours and central-area crop).

- `On_the_fly_loading.ipynb`  
  Explains the on-the-fly history construction approach used during training (history windows are built dynamically rather than stored as pre-expanded tensors).

- `splitting.py`  
  Produces a `splits.json` file defining the chronological **train/validation/test** partition of daily files (unit of split = whole days).

### Mask
Traffic4Cast provides a static mask in `BERLIN_static.h5` (and the reduced equivalent when produced). The mask is used to identify invalid cells so that training and evaluation can ignore locations that should not contribute to the loss/metrics.

## Land-use grid (`lbcs/`)

This folder contains scripts to generate an LBCS (Land-Based Classification Standards) grid aligned to the reduced Traffic4Cast raster. The process adapts code from a prior project to map OpenStreetMap (OSM) features into **nine high-level LBCS categories**, which are then used as context channels.

### Components
- `download_osm_berlin.py`  
  Downloads OSM data for the Berlin area. This is used instead of querying Overpass repeatedly when generating larger grids.

- `grid_maker/`  
  Utility package used to:
  1) generate a grid (H3 or square),
  2) map OSM tags to LBCS codes using a mapping table,
  3) compute per-cell code coverage (optionally with TF-IDF weighting),
  4) export GeoJSON and intermediate artifacts.

- `split_grid_processor.py` + `run_first_level_main.py`  
  Processes large GeoJSON inputs in tiles and extracts a matrix representation (`lbcs_matrix.json`) aligned with the reduced traffic grid. The pipeline executed by `run_first_level_main.py` produces the matrix used by the backend/training code.

### Resolution choice
The land-use grid have intentionally less resolution than the traffic grid (200 m cell size) to reduce noise from overly fine-grained OSM labels. The resulting tensor used by the model has **9 channels** and reduced spatial resolution relative to traffic (with scaling handled consistently in the shared code and metadata). Multiple tries with different resolutions ended up in the 1/2 one.

## Weather data (`weather/`)

This folder contains scripts and notebooks to clean and standardize meteorological data from the **Berlin–Tempelhof** weather station.

### Content
- `weather_cleaner.py`  
  Reads raw CSV files, selects the variables used by the project (precipitation, temperature, wind), and writes cleaned, synchronized records under:
  `hugging_face/weather_berlin_tempel/cleaned/`.

- `weather_data_analysis.ipynb`  
  Exploratory analysis, selection of `interest` ranges and validation of the weather series and selected variables.

### Temporal resolution and encoding
Weather records are aggregated at **10-minute** resolution. During training and inference, cleaned records are converted to a fixed-length weather vector by `shared/utils/weather_encoder.py`.

## Chronological splits and shuffling (`traffic/splitting.py`)

The dataset is split **by day** (whole HDF5 files) to reflect a forecasting setting:
- training uses 30 earlier days,
- validation uses later 7 unseen days,
- testing uses even later 7 days.

Shuffling is applied **only within the training loader** to support stochastic gradient optimization, while validation/test are kept chronological to measure generalization on temporally later data.
