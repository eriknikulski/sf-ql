from typing import SupportsFloat, Any

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import WrapperActType, ObsType, Wrapper
from minigrid.core.constants import DIR_TO_VEC
from minigrid.minigrid_env import MiniGridEnv

from wrappers.actions import Actions


class NonDirectionWrapper(Wrapper):
    """Changes direction dependent actions to basic right | down | left | up actions.

    Available actions are as follows:
    0 - right
    1 - down
    2 - left
    3 - up
    """
    def __init__(self, env: gym.Env):
        assert isinstance(env.unwrapped, MiniGridEnv), 'Base env needs to be a MiniGridEnv'
        super().__init__(env)

        self.minigrid_env: MiniGridEnv = env.unwrapped
        self.actions = Actions
        self.action_space = spaces.Discrete(len(self.actions))

    def step(
        self, action: WrapperActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if action not in self.actions:
            raise ValueError(f'Unknown action: {action}')

        terminated = False
        truncated = False

        # Get the position in front of the agent; based on minigrids env.front_pos()
        fwd_pos = self.minigrid_env.agent_pos + DIR_TO_VEC[action]

        # Get the contents of the cell in front of the agent
        fwd_cell = self.minigrid_env.grid.get(*fwd_pos)

        if fwd_cell is not None and not fwd_cell.can_overlap():
            # action is not possible -> just call Minigrids done action
            return self.env.step(self.minigrid_env.actions.done)

        # action is possible -> execute its components
        # turn in the right direction
        turn_reward = 0
        delta = action - self.minigrid_env.agent_dir
        # delta is 1 if action is just to the right or -3 if agent_dir is top and action right
        turn_right = delta == 1 or delta == -3

        if turn_right:
            # turn right once
            _obs, _reward, _terminated, _truncated, _info = self.env.step(self.minigrid_env.actions.right)
            turn_reward += _reward
            terminated = terminated or _terminated
            truncated = truncated or _truncated
        elif delta > 0:
            # turn left according to delta
            for _ in range(delta):
                _obs, _reward, _terminated, _truncated, _info = self.env.step(self.minigrid_env.actions.left)
                turn_reward += _reward
                terminated = terminated or _terminated
                truncated = truncated or _truncated

        # move
        obs, move_reward, _terminated, _truncated, info = self.env.step(self.minigrid_env.actions.forward)
        terminated = terminated or _terminated
        truncated = truncated or _truncated

        combined_reward = turn_reward + float(move_reward)
        return obs, combined_reward, terminated, truncated, info
