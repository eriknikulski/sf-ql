from typing import Optional, Tuple

import gymnasium as gym
import numpy as np

from sf_ql.config import Config
from sf_ql.utils.feature_extractor import MinigridFeaturesExtractor
from sf_ql.utils.logger import Logger


class QFunction:
    def __init__(
            self,
            feature_extractor: MinigridFeaturesExtractor,
            action_space_size: int,
            initial_q_value: float,
            alpha: float,
            gamma: float,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.action_space_size = action_space_size
        self.initial_q_value = initial_q_value
        self.alpha = alpha
        self.gamma = gamma

        self.z = None

    def __getitem__(self, key: Tuple[dict, int]) -> float:
        state, action = key
        return self.feature_extractor(state) @ self.z[action]

    def _init_z(self) -> None:
        self.z = np.full(
            shape=(self.action_space_size, self.feature_extractor.features_dim),
            fill_value=self.initial_q_value,
        )

    def tasks_init(self, tasks: int) -> None:
        pass

    def task_init(self, task: int, task_vec: np.ndarray) -> None:
        self._init_z()

    def step_init(self, state: dict, step: int) -> None:
        pass

    def update(
            self,
            state: dict,
            action: int,
            reward: float,
            next_state: dict,
            gamma: Optional[float] = None,
    ) -> None:
        gamma = gamma or self.gamma
        state_features = self.feature_extractor(state)
        next_state_features = self.feature_extractor(next_state)

        current_value = state_features @ self.z[action]
        next_state_q_table = self.z @ next_state_features
        max_next_value = np.max(next_state_q_table)

        td_target = reward + gamma * max_next_value
        td_error = td_target - current_value
        self.z[action] += self.alpha * td_error * state_features

    def get_action(self, state: dict) -> int:
        state_features = self.feature_extractor(state)
        state_q_table = self.z @ state_features

        # argmax tie-breaking
        action = int(np.argmax(state_q_table + 1e-8 * np.random.randn(len(state_q_table))))
        return action


class QL:
    def __init__(
            self,
            env: gym.Env,
            alpha: Optional[float] = None,
            epsilon: Optional[float] = None,
            epsilon_decay: Optional[float] = None,
            epsilon_lower_bound: Optional[float] = None,
            gamma: Optional[float] = None,
            initial_q_value: Optional[float] = None,
    ) -> None:
        self.config = Config()
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.env = env

        self.alpha = self.config.get_or_raise(alpha, 'q_learning', 'alpha')
        self.epsilon = self.config.get_or_raise(epsilon, 'q_learning', 'epsilon')
        self.epsilon_decay = self.config.get_or_raise(epsilon_decay, 'q_learning', 'epsilon_decay')
        self.epsilon_lower_bound = self.config.get_or_raise(epsilon_lower_bound, 'q_learning', 'epsilon_lower_bound')
        self.gamma = self.config.get_or_raise(gamma, 'q_learning', 'gamma')
        self.initial_q_value = self.config.get_or_raise(initial_q_value, 'q_learning', 'initial_q_value')

        self.use_sf_paper_reward = self.config.setting.use_sf_paper_reward

        self.Q = QFunction(
            feature_extractor=MinigridFeaturesExtractor(self.observation_space),
            action_space_size=self.action_space.n,
            initial_q_value=self.initial_q_value,
            alpha=self.alpha,
            gamma=self.gamma,
        )

        self.logger = Logger()

    def get_epsilon_greedy_action(self, state: dict, epsilon: Optional[float] = None) -> int:
        """
        Returns an epsilon-greedy action

        :param state: current state
        :param epsilon: exploration rate
        :return: action
        """
        epsilon = epsilon or self.epsilon
        select_random_action = np.random.binomial(n=1, p=epsilon)
        if select_random_action:
            action = self.env.action_space.sample()
        else:
            action = self.Q.get_action(state)
        return action

    def get_task_vec(self, task: int) -> np.ndarray:
        _, _ = self.env.reset(seed=task)
        if hasattr(self.env.unwrapped, 'rewards'):
            return np.append(self.env.unwrapped.rewards, 1)
        return np.array([1])

    def _learn_task(self, task: int, time_steps: int) -> None:
        """
        Learning function for Q-Learning

        :param task: task id
        :param time_steps: number of time steps
        :return:
        """
        task_vec = self.get_task_vec(task)
        if len(task_vec) > 1:
            self.logger.log_message(f'Task {task} has rewards {task_vec}')
        self.Q.task_init(task, task_vec=task_vec)

        steps_used = 0

        while steps_used < time_steps:

            # run episode
            state, _ = self.env.reset(seed=task)
            self.logger.new_episode()

            for episode_step in range(time_steps - steps_used):
                self.Q.step_init(state, steps_used)

                action = self.get_epsilon_greedy_action(state)

                next_state, reward, terminated, truncated, info = self.env.step(action)
                if self.use_sf_paper_reward:
                    reward = float(reward > 0)

                self.logger.log(reward=reward, epsilon=self.epsilon, alpha=self.alpha)

                gamma = 0 if terminated or truncated else self.gamma

                self.Q.update(state, action, float(reward), next_state, gamma=gamma)

                # decay parameter
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_lower_bound)

                state = next_state
                steps_used += 1

                if terminated or truncated or steps_used >= time_steps:
                    break

        self.logger.print_task_stats()

    def learn(self, tasks: int, time_steps_per_task: int) -> None:
        """
        Learning function for multitask Q-Learning

        :return:
        """
        self.Q.tasks_init(tasks)
        for task in range(tasks):
            self.logger.new_task(task)
            self._learn_task(task, time_steps_per_task)

    def predict(self, state: dict) -> tuple[int, None]:
        """
        Predict the action given a state

        :param state:
        :return: the model's action and the next hidden state (here None. This is to match stable_baselines3)
        """
        return self.Q.get_action(state), None
