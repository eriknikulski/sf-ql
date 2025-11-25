from typing import Optional

import gymnasium as gym
import numpy as np

from config import Config
from feature_extractor import MinigridFeaturesExtractor
from logger import Logger


class QFunction:
    def __init__(self, feature_extractor, z, alpha, gamma) -> None:
        self.feature_extractor = feature_extractor
        self.z = z

        self.alpha = alpha
        self.gamma = gamma

    def __getitem__(self, key) -> None:
        state, action = key
        state_features = self.feature_extractor(state)
        return state_features @ self.z[action]

    def update(self, state, action, reward, next_state, gamma: Optional[float] = None) -> None:
        gamma = gamma or self.gamma
        state_features = self.feature_extractor(state)
        next_state_features = self.feature_extractor(next_state)

        current_value = state_features @ self.z[action]
        next_state_q_table = self.z @ next_state_features
        max_next_value = np.max(next_state_q_table)

        td_target = reward + gamma * max_next_value
        td_error = td_target - current_value
        self.z[action] += self.alpha * td_error * state_features

    def get_action(self, state) -> int:
        state_features = self.feature_extractor(state)
        state_q_table = self.z @ state_features
        action = np.argmax(state_q_table)
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
            initial_z_value: Optional[float] = None,
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
        self.initial_z_value = self.config.get_or_raise(initial_z_value, 'q_learning', 'initial_z_value')

        feature_extractor = MinigridFeaturesExtractor(self.observation_space)
        z = np.full((self.action_space.n, feature_extractor.features_dim), fill_value=self.initial_z_value)

        self.Q = QFunction(
            feature_extractor=feature_extractor,
            z=z,
            alpha=self.alpha,
            gamma=self.gamma,
        )

        self.logger = Logger()

    def get_epsilon_greedy_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
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

    def learn_task(self, time_steps: int) -> None:
        """
        Learning function for Q-Learning

        :param time_steps: number of time steps
        :return:
        """
        new_episode = True
        state, _ = self.env.reset()

        for i in range(time_steps):
            gamma = self.gamma

            if new_episode:
                new_episode = False
                state, _ = self.env.reset()
                self.logger.new_episode()

            action = self.get_epsilon_greedy_action(state)

            next_state, reward, terminated, truncated, info = self.env.step(action)
            self.logger.log(reward=reward)

            if terminated or truncated:
                gamma = 0
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_lower_bound)
                new_episode = True

            self.Q.update(state, action, float(reward), next_state, gamma=gamma)

            state = next_state

        self.logger.print_stats()

    def learn(self, tasks: int, time_steps: int) -> None:
        """
        Learning function for multitask Q-Learning

        :param tasks: number of tasks
        :param time_steps: number of time steps
        :return:
        """
        for i in range(tasks):
            self.learn_task(time_steps)

    def predict(self, state) -> tuple[int, None]:
        """
        Predict the action given a state

        :param state:
        :return: the model's action and the next hidden state (here None. This is to match stable_baselines3)
        """
        return self.Q.get_action(state), None
