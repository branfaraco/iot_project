"""
Utility functions for the backend
--------------------------------

The :mod:`backend.utils` package contains small helper functions
required throughout the backend.  The most notable of these are the
data preprocessing routines used to transform raw history windows into
a flattened tensor format suitable for the neural network models.
"""

from .preprocess import prepare_inputs_raw, prepare_inputs_enriched  # noqa: F401

__all__ = [
    "prepare_inputs_raw",
    "prepare_inputs_enriched",
]