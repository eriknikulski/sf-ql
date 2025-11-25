import gymnasium as gym
from minigrid.wrappers import SymbolicObsWrapper

from q_learning import QL


def learn_policy(env: gym.Env, time_steps: int = 20_000):
    model = QL(env)
    model.learn(tasks=1, time_steps=time_steps)
    return model


def execute_policy(env: gym.Env, policy, time_steps: int = 1_000):
    observation, info = env.reset(seed=42)
    for _ in range(time_steps):
        action, _ = policy.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()


def main():
    env_id = 'MiniGrid-Empty-5x5-v0'
    # env_id = 'MiniGrid-FourRooms-v0'

    env = gym.make(env_id, render_mode='rgb_array')
    env = SymbolicObsWrapper(env)

    policy = learn_policy(env, time_steps=100_000)

    execute_policy(env, policy)

    env.close()


if __name__ == '__main__':
    main()
