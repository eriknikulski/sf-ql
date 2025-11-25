import gymnasium as gym
import minigrid.core.constants as minigrid_constants
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def rbf_features(state, grid_size=10, sigma=0.1):
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
    features = np.exp(-sq_norms / (sigma ** 2))
    return features


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    AGENT_VALUE = minigrid_constants.OBJECT_TO_IDX['agent']
    GOAL_VALUE = minigrid_constants.OBJECT_TO_IDX['goal']
    FEATURE_CONST = 1

    def __init__(self, observation_space: gym.Space) -> None:

        width, height = observation_space['image'].shape[:-1]
        self.grid_size = max(width, height)
        self.rbf_feature_grid_size = min(self.grid_size, 10)

        features_dim = self.rbf_feature_grid_size**2 + 2 + 1
        super().__init__(observation_space, features_dim)

    def forward(self, observations):
        idx = np.argwhere(observations['image'] == self.AGENT_VALUE)[0][:-1]
        norm_agent_position = observations['image'][*idx][:-1] / self.grid_size
        agent_pos_features = rbf_features(norm_agent_position, grid_size=self.rbf_feature_grid_size)

        agent_direction_features = minigrid_constants.DIR_TO_VEC[observations['direction']]

        return np.concatenate([[self.FEATURE_CONST], agent_pos_features, agent_direction_features])
