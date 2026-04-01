import numpy as np
from abc import ABC, abstractmethod

class AudioBackend(ABC):
    @abstractmethod
    def play(self, samples: np.ndarray, sample_rate: int) -> None: ...

    @abstractmethod
    def wait(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    def close(self) -> None:
        pass
