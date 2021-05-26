from abc import ABCMeta, abstractmethod

import numpy as np

from generic.optimization.model import OptimizationSense

_DEFAULT_ACCURACY = 1e-4


class TargetTracker(metaclass=ABCMeta):
    sense: OptimizationSense

    def __init__(self, *, sense, **kwargs):
        self.sense = sense

    @abstractmethod
    def __call__(self, current_value, **kwargs) -> float:
        pass

    @abstractmethod
    def reset(self):
        pass


class FixedTargetTracker(TargetTracker):
    overestimation_factor: float
    record_value: float

    def __init__(
        self,
        *,
        overestimation_factor: float = 0.1,
        min_overestimation_delta: float = 5 * _DEFAULT_ACCURACY,
        accuracy: float = _DEFAULT_ACCURACY,
        **kwargs
    ):
        assert overestimation_factor > 0, "Overestimation factor should be positive"
        assert min_overestimation_delta > 0, "Minimum overestimation factor should be positive"
        assert 0 < accuracy < 1, "Accuracy should be between 0 and 1"

        super().__init__(**kwargs)
        self.overestimation_factor = overestimation_factor
        self.min_overestimation_delta = min_overestimation_delta
        self.accuracy = accuracy
        self.reset()

    def __call__(self, current_value: float, **kwargs) -> float:
        if self.sense.value * (current_value - self.record_value) > 0:
            self.record_value = current_value

        overestimation_delta = max(
            self.min_overestimation_delta, (abs(self.record_value) + self.accuracy) * self.overestimation_factor
        )
        target = self.record_value + self.sense.value * overestimation_delta
        return target

    def reset(self):
        self.record_value = -self.sense * np.infty
