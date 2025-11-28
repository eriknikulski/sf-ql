from typing import Optional

import gymnasium as gym
import minigrid.core.constants as minigrid_constants
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from config import Config


def rbf_features(state, grid_size: int = 10, sigma: float = 0.1):
    """
    state: 1D array of length 2 (x,y)
    grid_size: number of centers per dimension
    sigma: width of Gaussian
    """
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    centers = np.stack([xx.ravel(), yy.ravel()], axis=1)  # shape (grid_size^2, 2)

    # compute RBF outputs
    diffs = centers - state
    sq_norms = np.sum(diffs ** 2, axis=1)
    features = np.exp(-sq_norms / sigma)
    return features


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    AGENT_VALUE = minigrid_constants.OBJECT_TO_IDX['agent']
    GOAL_VALUE = minigrid_constants.OBJECT_TO_IDX['goal']
    FEATURE_CONST = 1
    BIAS_DIM = 1
    AGENT_DIRECTION_DIMS = 2

    def __init__(
            self,
            observation_space: gym.Space,
            max_rbf_grid_size: Optional[int] = None,
            rbf_sigma: Optional[float] = None,
            include_goal: Optional[bool] = None,
            include_direction: Optional[bool] = None,
    ) -> None:
        self.config = Config()

        self.max_rbf_grid_size = self.config.get_or_raise(max_rbf_grid_size, 'feature_extractor', 'max_rbf_grid_size')
        self.rbf_sigma = self.config.get_or_raise(rbf_sigma, 'feature_extractor', 'rbf_sigma')
        self.include_goal = self.config.get_or_raise(include_goal, 'feature_extractor', 'include_goal')
        self.include_direction = self.config.get_or_raise(include_direction, 'feature_extractor', 'include_direction')

        self.grid_size = observation_space['image'].shape[:-1]
        self.rbf_grid_size = min(*self.grid_size, self.max_rbf_grid_size)

        # calculate feature dimension
        rbf_feature_dims = self.rbf_grid_size ** 2
        if self.include_goal:
            rbf_feature_dims *= 2
        direction_feature_dims = self.AGENT_DIRECTION_DIMS if self.include_direction else 0
        features_dim = rbf_feature_dims + direction_feature_dims + self.BIAS_DIM

        super().__init__(observation_space, features_dim)

    def get_features(
            self,
            observations: dict,
            obj_idx: int,
            default_position: Optional[np.ndarray] = None,
    ) -> np.ndarray | None:
        matches = np.argwhere(observations['image'] == obj_idx)

        if len(matches) > 0:
            position = matches[0][-1]
        else:
            position = default_position

        if position is None:
            return None

        normalized_position = position / self.grid_size
        pos_features = rbf_features(normalized_position, grid_size=self.rbf_grid_size, sigma=self.rbf_sigma)

        return pos_features

    def forward(self, observations: dict) -> np.ndarray:
        # agent features
        agent_pos = self.get_features(observations, self.AGENT_VALUE)

        # agent direction features
        if self.include_direction:
            agent_direction = minigrid_constants.DIR_TO_VEC[observations['direction']]
        else:
            agent_direction = np.array([])

        # goal features
        if self.include_goal:
            goal_pos = self.get_features(observations, self.GOAL_VALUE)
            if goal_pos is None:
                goal_pos = agent_pos
        else:
            goal_pos = np.array([])

        # combined features
        features = np.concatenate([
            [self.FEATURE_CONST],
            agent_pos,
            agent_direction,
            goal_pos,
        ])

        return features
