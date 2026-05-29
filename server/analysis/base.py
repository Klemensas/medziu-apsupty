"""Base protocols for the analysis pipeline.

Processors analyse raw tracking results (landmarks) and produce typed
analysis dataclasses.  Transformers consume one or more analysis results
and map them to downstream effects (visuals, audio, OSC, …).

Each transformer returns a JSON-serialisable dict from ``transform()``
so the pipeline can bundle all outputs and send them to the feed.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class Processor(Protocol):
    """Turns raw tracking landmarks into an analysis dataclass."""

    @property
    def input_type(self) -> type:
        """The tracking-result type this processor handles."""
        ...

    def process(self, data: Any) -> Any:
        """Analyse *data* and return an analysis dataclass."""
        ...


class Transformer(Protocol):
    """Consumes analysis results and produces an output for the feed."""

    @property
    def name(self) -> str:
        """Unique key used in the output dict sent to the feed."""
        ...

    @property
    def input_types(self) -> frozenset[type]:
        """Set of analysis types this transformer wants to receive."""
        ...

    def transform(self, analyses: dict[type, Any]) -> dict[str, Any]:
        """Process analyses and return a JSON-serialisable output dict.

        The returned dict is included in the server response under
        ``outputs[self.name]`` so the feed can act on it.
        """
        ...

    def debug(self, frame_bgr: np.ndarray) -> None:
        """Draw a debug overlay onto *frame_bgr* in-place."""
        ...
