from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import pi
from typing import Deque

import numpy as np


def wrap_angle(angle_radians: float) -> float:
    return float((angle_radians + pi) % (2.0 * pi) - pi)


def turns_to_radians(turns: float) -> float:
    return float(turns * 2.0 * pi)


def radians_to_turns(radians: float) -> float:
    return float(radians / (2.0 * pi))


def normalize_value(value: float, scale: float) -> float:
    if scale <= 0.0:
        return 0.0
    return float(np.clip(value / scale, -1.0, 1.0))


def build_features(
    position_turns: float,
    velocity_turns_per_s: float,
    torque_nm: float,
    velocity_scale_turns_per_s: float,
    torque_scale_nm: float,
) -> np.ndarray:
    theta = turns_to_radians(position_turns)
    return np.array(
        [
            np.sin(theta),
            np.cos(theta),
            normalize_value(velocity_turns_per_s, velocity_scale_turns_per_s),
            normalize_value(torque_nm, torque_scale_nm),
        ],
        dtype=np.float32,
    )


@dataclass
class ObservationBuilder:
    history_length: int
    velocity_scale_turns_per_s: float
    torque_scale_nm: float

    def __post_init__(self) -> None:
        self._history: Deque[np.ndarray] = deque(maxlen=self.history_length)

    def reset(
        self,
        position_turns: float,
        velocity_turns_per_s: float,
        torque_nm: float,
    ) -> np.ndarray:
        feature = build_features(
            position_turns,
            velocity_turns_per_s,
            torque_nm,
            self.velocity_scale_turns_per_s,
            self.torque_scale_nm,
        )
        self._history = deque((feature.copy() for _ in range(self.history_length)), maxlen=self.history_length)
        return self.flatten()

    def push(
        self,
        position_turns: float,
        velocity_turns_per_s: float,
        torque_nm: float,
    ) -> np.ndarray:
        feature = build_features(
            position_turns,
            velocity_turns_per_s,
            torque_nm,
            self.velocity_scale_turns_per_s,
            self.torque_scale_nm,
        )
        self._history.append(feature)
        while len(self._history) < self.history_length:
            self._history.appendleft(feature.copy())
        return self.flatten()

    def flatten(self) -> np.ndarray:
        if not self._history:
            raise RuntimeError("ObservationBuilder has not been reset")
        return np.concatenate(list(self._history), axis=0).astype(np.float32, copy=False)
