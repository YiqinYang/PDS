from typing import Dict

import flax.linen as nn
import gym
import numpy as np


def evaluate(agent: nn.Module, env: gym.Env,
             num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': []}

    for _ in range(num_episodes):
        observation, done = env.reset(), False

        # r_return = 0
        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, reward, done, info = env.step(action)
            # r_return += reward
        # print(r_return)
        for k in stats.keys():
            stats[k].append(info['episode'][k])

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats
