# Shared library

The `shared/` package contains code that is reused across the
preprocessing, training and inference stages.  It provides model
definitions, data handling utilities and common training routines.
Rather than duplicating logic in the training scripts and backend,
these functions live here and are imported as needed.

## Subpackages and modules

- `models/` – Defines the U‑Net architectures used in this project.
  - `unet_base.py`: A simplified U‑Net that consumes a flattened
    traffic history `(H·8, 264, 432)` and outputs `F=4` future
    frames of aggregated volumes.
  - `unet_film.py`: A U‑Net variant with FiLM conditioning.  It
    uses land‑use channels and a weather embedding and
    applies channel‑wise affine transformations based on the weather
    vector.
- `utils/` – A collection of helpers for data processing and training:
  - `data_reduction.py`: Functions to crop Traffic4Cast frames to the
    08:20–18:20 window, select the 264×432 spatial region and
    normalise the eight channels.  These functions are called both
    offline (to create the reduced dataset) and online (by the
    backend).
  - `weather_encoder.py`: Encodes the cleaned weather variables into
    a fixed‑length vector.  During inference the backend calls
    `WeatherEncoder.encode_from_timestamp()` to convert a timestamp
    into the appropriate embedding.
  - `lbcs.py`: Loads the one‑hot land‑use matrices and provides
    functions to align them to the traffic grid.  The nine LBCS
    channels are repeated across the history dimension and concatenated
    with traffic channels.
  - `mask.py`: Loads the road mask from `BERLIN_static.h5` and
    returns binary arrays marking valid road cells; used by the
    custom loss.
  - `losses.py`: Implements the `MaskedMAEFocalLoss`, which
    combines an MAE on valid road cells with a focal weighting.
  - `indexing.py` and `lbcs.py`: Helpers for computing history
    indices and sampling windows.

