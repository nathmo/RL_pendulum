from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PendulumFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, hidden_size: int = 128) -> None:
        super().__init__(observation_space, features_dim=hidden_size)
        input_dim = int(np.prod(observation_space.shape))
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.network(observations)


def build_policy_kwargs(hidden_size: int = 128, head_size: int = 64) -> dict[str, Any]:
    return {
        "features_extractor_class": PendulumFeatureExtractor,
        "features_extractor_kwargs": {"hidden_size": hidden_size},
        "net_arch": {"pi": [head_size], "vf": [head_size]},
        "activation_fn": nn.Tanh,
        "ortho_init": True,
    }


class ExportedPendulumPolicy(nn.Module):
    def __init__(self, policy: Any) -> None:
        super().__init__()
        self.features_extractor = deepcopy(policy.features_extractor).eval()
        self.policy_net = deepcopy(policy.mlp_extractor.policy_net).eval()
        self.action_net = deepcopy(policy.action_net).eval()

    def forward(self, obs: th.Tensor) -> th.Tensor:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        obs = obs.float()
        features = self.features_extractor(obs)
        latent_pi = self.policy_net(features)
        action = th.tanh(self.action_net(latent_pi))
        return action


def export_policy_to_onnx(policy: Any, export_path: str | Path, sample_obs: np.ndarray, opset: int) -> None:
    module = ExportedPendulumPolicy(policy).eval().cpu()
    example = th.as_tensor(sample_obs[None, :], dtype=th.float32)
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    th.onnx.export(
        module,
        example,
        str(export_path),
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
