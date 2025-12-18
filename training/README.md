
## Training philosophy

The raw Traffic4Cast data are large and sparse; storing all possible
history tensors on disk would be prohibitively expensive.  Instead,
histories are constructed **on the fly** during training: each batch
samples a random day from the training split, loads the cropped frames
into memory and slices out `H = 12` consecutive frames for the input and `F = 4` frames for
the target.  
## Scripts

The following scripts implement the two training regimes:

- `unet_base/training.py` – Trains the baseline U‑Net on traffic
  data only.  It uses the `TrafficDataset` class to load cropped
  frames from `DATA_ROOT/BERLIN_reduced/traffic_data/`, builds
  input histories on the fly and evaluates on the validation days.
  Training and validation loss curves are saved as JSON files for
  later visualisation.
- `unet_sensor/training.py` – Trains the enriched U‑Net with FiLM
  conditioning.  In addition to the traffic history, it loads the
  one‑hot LBCS tensor (nine channels) from `LBCS_PATH` and encodes
  the current weather variables using `WeatherEncoder`.  The LBCS
  channels are repeated across the history dimension and concatenated
  with the traffic channels to form an input of shape
  `(H·(8+9), 264, 432)`.  The weather embedding is passed
  through FiLM layers during inference.

Both scripts read the training/validation/test splits from
`DATA_ROOT/BERLIN_reduced/splits/splits.json`.  They also expect the environment
variables defined in the `.env` file (e.g. `DATA_ROOT`, `LBCS_PATH`,
`WEATHER_ROOT` and `MODEL_PARAMETERS_DIR`).


