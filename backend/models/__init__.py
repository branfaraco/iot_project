"""
Model loading utilities for the backend.

Import `load_raw_model` and `load_enriched_model` from this module
to instantiate the pre-trained networks used for real-time
inference. These functions wrap the creation of the appropriate
model classes (defined in the shared.models package) and
load weights from disk.
"""

from .raw_model import load_raw_model
from .enriched_model import load_enriched_model

__all__ = ["load_raw_model", "load_enriched_model"]