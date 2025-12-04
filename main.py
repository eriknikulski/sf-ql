import gymnasium as gym

import simple_minigrid
from simple_minigrid.wrappers import SymbolicObsWrapper

from sf_ql.config import Config
from sf_ql.algorithms import QL
from sf_ql.algorithms import SFQL


def learn_policy(env: gym.Env, tasks: int, time_steps_per_task: int):
    model = SFQL(env)
    model.learn(tasks=tasks, time_steps_per_task=time_steps_per_task)
    return model


def main():
    config = Config()

    env_id = config.setting.env_id
    render_mode = config.setting.render_mode
    max_episode_steps = config.setting.max_episode_steps

    env = gym.make(env_id, max_episode_steps=max_episode_steps, render_mode=render_mode)
    env = SymbolicObsWrapper(env)

    policy = learn_policy(
        env,
        tasks=config.setting.tasks,
        time_steps_per_task=config.setting.time_steps_per_task,
    )

    env.close()


if __name__ == '__main__':
    main()
