import gymnasium as gym
from minigrid.wrappers import SymbolicObsWrapper

from config import Config
from logger import Logger
from q_learning import QL


def learn_policy(env: gym.Env, tasks: int, time_steps_per_task: int):
    model = QL(env, tasks=tasks, time_steps_per_task=time_steps_per_task)
    model.learn()
    return model


def eval_policy(env: gym.Env, policy, time_steps: int):
    logger = Logger(log_interval=None)
    logger.log_message('\nStarting evaluation...')
    logger.new_episode()

    # TODO: think about which task the evaluation should run
    observation, info = env.reset(seed=42)

    for _ in range(time_steps):
        action, _ = policy.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        logger.log(reward=reward)

        if terminated or truncated:
            observation, info = env.reset()
            logger.new_episode()

    logger.print_stats()


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

    eval_steps = config.setting.eval_steps
    eval_policy(env, policy, time_steps=eval_steps)

    env.close()


if __name__ == '__main__':
    main()
