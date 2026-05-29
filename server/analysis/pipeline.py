"""Analysis pipeline — routes tracking results through processors and
forwards the produced analyses to registered transformers.

Typical usage::

    from analysis import Pipeline, HandProcessor, FaceProcessor
    from analysis.output import BaseBeatTransformer

    pipeline = Pipeline()
    pipeline.register_processor(HandProcessor())
    pipeline.register_processor(FaceProcessor())
    pipeline.register_transformer(BaseBeatTransformer())

    outputs = pipeline.process(hand_result, face_result)
    # outputs == {"beat": {"active": True, "notes": [...], ...}}
"""

from __future__ import annotations

import logging
from typing import Any

from .base import Processor, Transformer

log = logging.getLogger(__name__)


class Pipeline:
    """Central coordinator that owns processors and transformers.

    Processors are keyed by the tracking-result *type* they accept.
    Transformers declare which *analysis* types they consume; the pipeline
    delivers matching subsets after every processing tick.

    ``process()`` returns a ``{transformer.name: output_dict}`` mapping
    ready for JSON serialisation and delivery to the feed.
    """

    def __init__(self) -> None:
        self._processors: dict[type, Processor] = {}
        self._transformers: list[Transformer] = []

    def register_processor(self, processor: Processor) -> None:
        self._processors[processor.input_type] = processor

    def register_transformer(self, transformer: Transformer) -> None:
        self._transformers.append(transformer)

    def process(self, *tracking_results: Any) -> dict[str, Any]:
        """Run one pipeline tick.

        1. Route each tracking result to its registered processor.
        2. Collect the analysis dataclasses keyed by their type.
        3. Fan out matching analyses to every registered transformer.
        4. Collect each transformer's returned output dict.

        Returns ``{transformer_name: output_dict}`` — the combined
        payload that the server sends to the feed alongside video.
        """
        analyses: dict[type, Any] = {}

        for result in tracking_results:
            processor = self._processors.get(type(result))
            if processor is None:
                continue
            analysis = processor.process(result)
            analyses[type(analysis)] = analysis

        outputs: dict[str, Any] = {}
        for transformer in self._transformers:
            relevant = {
                k: v for k, v in analyses.items() if k in transformer.input_types
            }
            if relevant:
                output = transformer.transform(relevant)
                if output is not None:
                    outputs[transformer.name] = output

        return outputs
