import numpy as np
import jax.numpy as jnp
import torch
import time
from torch import nn
import torch.nn.functional as F
from preprocessing.analysis import correlation
from dataset_utils import *
import matplotlib.pyplot as plt
MIDDLE_SHAPE = 256


class RandomReward(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, MIDDLE_SHAPE)
        self.l2 = nn.Linear(MIDDLE_SHAPE, MIDDLE_SHAPE)
        # self.l2_2 = nn.Linear(256, 256)
        # self.l2_3 = nn.Linear(256, 256)
        self.l3 = nn.Linear(MIDDLE_SHAPE, 1)

        self.al1 = nn.Linear(action_dim, MIDDLE_SHAPE)
        self.al2 = nn.Linear(MIDDLE_SHAPE, MIDDLE_SHAPE)
        self.al3 = nn.Linear(MIDDLE_SHAPE, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        # q1 = F.relu(self.l2_2(q1))
        q1 = self.l3(q1)
        return q1
        # c1 = F.relu(self.al1(action))
        # c2 = F.relu(self.al2(c1))
        # c3 = self.al3(c2)
        # return q1 + c3

    def load(self, filename):
        self.partial_load(self, torch.load(filename + "_critic"),
                          non_load_names=["l6.weight", "l6.bias", "l3.weight", "l3.bias"])

    def partial_load(self, network, state_dict, non_load_names=[]):

        own_state = network.state_dict()
        for name, param in state_dict.items():
            if name not in own_state or name in non_load_names:
                continue
            our_param = own_state[name]
            if our_param.shape != param.shape:
                print(f"{name} shape Mismatch, did not load, shape:{our_param.shape} {param.shape}")
                continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
            print(f"Successful Load:{name} ")


def reward_gen(replay_buffer, state_dim, action_dim, reward_dim=100, load_model=None):
    print("Begin Creating Reward ...")
    batch_size = 256
    torch.manual_seed(int(time.time()))
    random_reward_net = [RandomReward(state_dim, action_dim) for _ in range(reward_dim)]
    if load_model is not None:
        for net in random_reward_net:
            net.load(f"./models/{load_model}")
    # random_reward_net = [RandomReward(state_dim, action_dim) for _ in range(max(256, reward_dim))]
    for net in random_reward_net:
        net.cuda()
    state = torch.tensor(replay_buffer.observations).cuda()
    # next_state = torch.tensor(replay_buffer.next_state).cuda()
    action = torch.tensor(replay_buffer.actions).cuda()
    buffer_size = len(replay_buffer.observations)
    random_rewards = np.zeros((buffer_size, reward_dim))
    # random_rewards = np.zeros((buffer_size, max(256, reward_dim)))
    for i in range(buffer_size // batch_size + 1):
        start, end = i * batch_size, min((i + 1) * batch_size, buffer_size)
        random_reward = [net(state[start:end], action[start:end]).cpu().detach().numpy() for net
                         in random_reward_net]
        random_rewards[start:end, :] = np.array(random_reward).T
    del random_reward_net
    return random_rewards


def reward_randomization_nn(replay_buffer, state_dim, action_dim, reward_dim=100, scale=1,
                            load_model=None, reward_type=None):
    if reward_type == "none" or reward_type == "baseline" or reward_type is None:
        return
    if reward_type == "zero":
        replay_buffer.rewards = np.zeros_like(replay_buffer.rewards)
        return
    random_rewards = reward_gen(replay_buffer, state_dim, action_dim, reward_dim, load_model)
    coe = correlation(random_rewards, replay_buffer.rewards)
    print(coe)
    print("max, mean, min, variance")
    print(np.max(coe), np.mean(coe), np.min(coe), np.var(coe))
    coe_index = np.argsort(coe)
    if reward_type == "max":
        coe_index = coe_index[::-1]
    else:
        assert reward_type == "min"
    random_rewards = random_rewards[:, coe_index]
    random_rewards -= np.mean(random_rewards, axis=0)
    random_rewards /= np.std(random_rewards, axis=0)
    random_rewards *= np.std(replay_buffer.rewards)
    random_rewards += np.mean(replay_buffer.rewards)
    diff = replay_buffer.rewards - random_rewards[:, 0]
    trajs = split_into_trajectories(replay_buffer.observations, replay_buffer.actions, diff, replay_buffer.masks,
                                    replay_buffer.dones_float, replay_buffer.next_observations)
    replay_buffer.rewards = random_rewards[:, 0] * scale
    # replay_buffer.rewards = np.zeros_like(replay_buffer.rewards)
    print("Finish Creating Reward")
