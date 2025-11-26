import logging
from dataclasses import dataclass

import numpy as np
from typing import SupportsFloat, Optional

from config import Config


@dataclass
class EpisodeStats:
    episode_reward: float = 0
    steps: int = 0


class Logger:
    def __init__(self, name: str = __name__, log_interval: Optional[int] = None, level: Optional[int] = None) -> None:
        self.config = Config()
        self.stats = []
        self.log_interval = self.config.get_or_raise(log_interval, 'logger', 'log_interval')
        self.level = self.config.get_or_raise(level, 'logger', 'level')

        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)

    def new_episode(self) -> None:
        if self.stats and self.log_interval is not None and len(self.stats) % self.log_interval == 0:
            self.print_stats(last_n=self.log_interval)

        self.stats.append(EpisodeStats())

    def log(self, reward: SupportsFloat) -> None:
        if not self.stats:
            raise RuntimeError('No episode started. Call new_episode() before logging rewards.')
        self.stats[-1].episode_reward += reward
        self.stats[-1].steps += 1

    def print_stats(self, last_n: Optional[int] = None) -> None:
        stats = self.stats[-last_n:] if last_n else self.stats

        # remove an episode that was just started without any steps in it
        if stats and stats[-1].steps == 0:
            stats = stats[:-1]

        n = len(stats)

        preamble = '\n' if n == len(self.stats) else ''
        mean_episode_length = sum(entry.steps for entry in stats) / n
        median_episode_length = np.median(np.array([entry.steps for entry in stats]))
        mean_episode_reward = sum(entry.episode_reward for entry in stats) / n

        self.logger.info(f'{preamble}'
              f'Last {n} episodes: '
              f'Mean length {mean_episode_length:.2f} | '
              f'Median length {median_episode_length:.2f} | '
              f'Mean reward {mean_episode_reward:.3f}')

    def log_message(self, message: str) -> None:
        self.logger.info(message)
