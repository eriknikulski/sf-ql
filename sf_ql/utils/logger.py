import logging
from dataclasses import dataclass, field

import numpy as np
from typing import SupportsFloat, Optional, List

from tensorboard import program
from torch.utils.tensorboard import SummaryWriter

from sf_ql.config import Config


@dataclass
class EpisodeStats:
    rewards: List[float] = field(default_factory=list)
    steps: int = 0

    @property
    def return_(self) -> float:
        return sum(self.rewards)


@dataclass
class TaskStats:
    id: int
    episodes: List[EpisodeStats] = field(default_factory=list)

    @property
    def return_(self) -> float:
        return sum(episode.return_ for episode in self.episodes)


@dataclass
class Stats:
    tasks: List[TaskStats] = field(default_factory=list)

    @property
    def return_(self) -> float:
        return sum(task.return_ for task in self.tasks)

    @property
    def episodes(self) -> int:
        return sum(len(task.episodes) for task in self.tasks)

    @property
    def avg_return(self) -> float:
        return sum(episode.return_ for task in self.tasks for episode in task.episodes) / self.episodes


class Logger:
    def __init__(
            self,
            name: str = __name__,
            log_interval: Optional[int] = None,
            level: Optional[int] = None,
            evaluation: bool = False,
            autostart_tensorboard: bool = False,
    ) -> None:
        self.config = Config()
        self.log_interval = log_interval or self.config.logger.log_interval     # None is a valid value
        self.level = self.config.get_or_raise(level, 'logger', 'level')
        self.evaluation = evaluation
        self.autostart_tensorboard = autostart_tensorboard or self.config.logger.autostart_tensorboard

        self.stats: Stats = Stats()
        self.task: TaskStats | None = None
        self.episode: EpisodeStats | None = None
        self.global_step = 0

        # set up logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)

        # set up tensorboard
        if not self.evaluation:
            if self.autostart_tensorboard:
                Logger.start_tensorboard()
            self.tb_writer = SummaryWriter()

    @staticmethod
    def start_tensorboard() -> None:
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', 'runs', '--port', '6006'])
        url = tb.launch()
        print(f'TensorBoard running at: {url}')

    def new_task(self, task_id: int) -> None:
        task_stats = TaskStats(task_id)
        self.stats.tasks.append(task_stats)
        self.task = task_stats

    def new_episode(self) -> None:
        if (
                len(self.task.episodes) > 0
                and self.log_interval is not None
                and len(self.task.episodes) % self.log_interval == 0
        ):
            self.print_task_stats(last_episodes=self.log_interval)

        episode_stats = EpisodeStats()
        self.task.episodes.append(episode_stats)
        self.episode = episode_stats

    def log(self, reward: SupportsFloat, **kwargs) -> None:
        if not self.task or not self.episode:
            raise RuntimeError('No task or episode started. '
                               'Make sure to call new_task() and new_episode() before logging rewards.')

        self.episode.rewards.append(float(reward))
        self.episode.steps += 1
        self.global_step += 1

        if not self.evaluation:
            self.tb_writer.add_scalar('reward', reward, self.global_step)
            self.tb_writer.add_scalar('task_return', self.task.return_, self.global_step)
            self.tb_writer.add_scalar('avg_return', self.stats.avg_return, self.global_step)

            for key, value in kwargs.items():
                self.tb_writer.add_scalar(key, value, self.global_step)

    def print_task_stats(self, last_episodes: Optional[int] = None) -> None:
        stats = self.task.episodes[-last_episodes:] if last_episodes else self.task.episodes

        # remove an episode that was just started without any steps in it
        if stats and stats[-1].steps == 0:
            stats = stats[:-1]

        n = len(stats)

        preamble = '\n' if n == len(self.task.episodes) else ''

        steps = np.array([episode.steps for episode in stats])
        mean_episode_length = np.mean(steps)
        std_episode_length = np.std(steps)
        median_episode_length = np.median(steps)
        mean_episode_reward = sum(sum(episode.rewards) for episode in stats) / n

        self.logger.info(f'{preamble}'
              f'Last {n} episodes: '
              f'Mean length {mean_episode_length:.2f}±{std_episode_length:.2f} | '
              f'Median length {median_episode_length:.2f} | '
              f'Mean reward {mean_episode_reward:.3f}')

    def log_message(self, message: str) -> None:
        self.logger.info(message)
