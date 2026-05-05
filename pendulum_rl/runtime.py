from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


class OnnxPendulumPolicy:
    def __init__(self, model_path: str | Path, providers: list[str] | None = None) -> None:
        self.model_path = str(model_path)
        self.session = ort.InferenceSession(self.model_path, providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, obs: np.ndarray) -> float:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        action = self.session.run([self.output_name], {self.input_name: obs})[0]
        return float(np.clip(action.reshape(-1)[0], -1.0, 1.0))
