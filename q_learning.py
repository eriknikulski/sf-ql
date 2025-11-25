from typing import Optional, SupportsFloat, Union

import gymnasium as gym
import numpy as np

from feature_extractor import MinigridFeaturesExtractor


class Logger:
    def __init__(self, log_interval: Union[int, None] = 100) -> None:
        self.stats = []
        self.log_interval = log_interval

    def new_episode(self) -> None:
        if self.stats and self.log_interval is not None and len(self.stats) % self.log_interval == 0:
            self.print_episode_stats()

        self.stats.append({
            'episode_reward': 0,
            'steps': 0,
        })

    def log(self, reward: SupportsFloat) -> None:
        self.stats[-1]['episode_reward'] += reward
        self.stats[-1]['steps'] += 1

    def print_episode_stats(self, index: int = -1) -> None:
        n = len(self.stats)
        episode_length = self.stats[index]['steps']
        episode_reward = self.stats[index]['episode_reward']
        print(f'Episode: {n} | Length {episode_length} | Reward {episode_reward:.3f}')

    def print_stats(self):
        _len = len(self.stats)
        mean_episode_length = sum(entry['steps'] for entry in self.stats) / _len
        mean_episode_reward = sum(entry['episode_reward'] for entry in self.stats) / _len
        mean_episode_step_reward = sum(entry['episode_reward'] / entry['steps'] for entry in self.stats) / _len
        print(f'\n\n'
              f'{_len} episodes | '
              f'Mean length {mean_episode_length} | '
              f'Mean reward {mean_episode_reward:.3f} | '
              f'Mean avg step reward {mean_episode_step_reward:.3f}')


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
            alpha: float = 0.1,
            epsilon: float = 0.15,
            epsilon_decay: float = 1.0,
            epsilon_lower_bound: float = 0.1,
            gamma: float = 0.95,
            initial_z_value: float = 1e-4,
    ) -> None:
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.env = env

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_lower_bound = epsilon_lower_bound

        feature_extractor = MinigridFeaturesExtractor(self.observation_space, rbf_max_grid_size=10)
        z = np.full((self.action_space.n, feature_extractor.features_dim), fill_value=initial_z_value)

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
