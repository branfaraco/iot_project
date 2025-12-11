"""
Inference subpackage for the backend.

This package contains modules for streaming data from the test
dataset and for orchestrating model inference on that data. The
`stream` module defines how to iterate over the test split, while
`pipeline` defines a simple dual‑model inference pipeline that can
process each sample and return predictions and metadata.
"""

from .stream import DatasetIterator, dataset_stream
from .pipeline import InferencePipeline

__all__ = ["DatasetIterator", "dataset_stream", "InferencePipeline"]