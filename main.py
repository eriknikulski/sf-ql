import gymnasium as gym
from minigrid.wrappers import SymbolicObsWrapper

from logger import Logger
from q_learning import QL


def learn_policy(env: gym.Env, time_steps: int):
    model = QL(env)
    model.learn(tasks=1, time_steps=time_steps)
    return model


def eval_policy(env: gym.Env, policy, time_steps: int):
    logger = Logger(log_interval=None)
    logger.log_message('\nStarting evaluation...')
    logger.new_episode()

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
    env_id = 'MiniGrid-Empty-5x5-v0'
    # env_id = 'MiniGrid-FourRooms-v0'

    env = gym.make(env_id, render_mode='rgb_array')
    env = SymbolicObsWrapper(env)

    policy = learn_policy(env, time_steps=100_000)

    eval_policy(env, policy, time_steps=1_000)

    env.close()


if __name__ == '__main__':
    main()
