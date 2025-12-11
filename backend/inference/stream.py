
import asyncio
from typing import Iterable, Dict, Any


class DatasetIterator:
    """
    Simple iterator over an iterable dataset.

    This class wraps any iterable (e.g. a list of samples or a
    generator) and exposes a `next` method that returns the next
    element, looping back to the start when the iterator is
    exhausted. This allows for continuous streaming of finite
    datasets.
    """
    def __init__(self, data: Iterable[Dict[str, Any]]):
        self._data = list(data)
        self._index = 0
        if not self._data:
            raise ValueError("DatasetIterator: data must contain at least one sample")

    def next(self) -> Dict[str, Any]:
        item = self._data[self._index]
        self._index = (self._index + 1) % len(self._data)
        return item


async def dataset_stream(iterator: DatasetIterator, delay: float = 0.2):
    """
    Asynchronous generator that yields one dataset sample at a fixed
    interval. The `delay` parameter controls the time in seconds
    between consecutive samples. Use this in the WebSocket handler
    to push predictions to the frontend at a user‑configured speed.
    """
    while True:
        yield iterator.next()
        await asyncio.sleep(delay)