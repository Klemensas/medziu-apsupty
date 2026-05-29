from abc import ABC, abstractmethod

import numpy as np


class AudioBackend(ABC):
    @abstractmethod
    def play(self, samples: np.ndarray, sample_rate: int) -> None: ...

    @abstractmethod
    def wait(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    def close(self) -> None:  # noqa: B027
        pass
