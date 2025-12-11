"""
Backend package
================

This package contains the implementation of the data streaming
services (traffic and weather generators) as well as the core
backend API that consumes those streams, runs the pre‑trained
models and exposes predictions via a WebSocket interface.  The
components are designed to be assembled together according to the
architecture described in the provided specification.

The backend makes use of a number of utilities defined in the
``gpt`` package.  Most notably it relies on the following modules:

* :mod:`gpt.inference.pipeline` – wraps the raw and enriched models
  into a simple inference pipeline.
* :mod:`gpt.models.raw_model` and :mod:`gpt.models.enriched_model` –
  functions for constructing and loading the baseline and FiLM
  models.
* :mod:`gpt.shared.utils.losses` – provides the
  ``MaskedMAEFocalLoss`` criterion.
* :mod:`gpt.shared.utils.lbcs` and :mod:`gpt.shared.utils.mask` –
  used to load the static land use and road masks.
* :mod:`gpt.shared.utils.weather_encoder` – optionally used to
  normalise raw weather observations into a fixed‑length vector.

The ``backend.service`` module instantiates the models, starts
background tasks to consume the data streams and exposes a FastAPI
application with the `/stream` WebSocket and `/metrics/loss_history`
HTTP endpoints.  See the individual modules for further details.

This package is not meant to be executed directly; instead, run the
individual services (traffic generator, weather generator and
backend) with an ASGI server such as ``hypercorn`` or ``uvicorn``.

Example usage::

    # Run the traffic generator on port 8001
    hypercorn backend.traffic_generator:app --bind 0.0.0.0:8001

    # Run the weather generator on port 8002
    hypercorn backend.weather_generator:app --bind 0.0.0.0:8002

    # Run the backend on port 8000
    hypercorn backend.service:app --bind 0.0.0.0:8000

"""

__all__ = [
    "traffic_generator",
    "weather_generator",
    "service",
]