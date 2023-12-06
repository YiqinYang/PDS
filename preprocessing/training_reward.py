import torch
import torch.utils.data.dataset as dataset
import numpy as np
import tqdm
from torch import nn
import torch.nn.functional as F
from preprocessing.analysis import correlation
from dataset_utils import *
import matplotlib.pyplot as plt
import torch.optim as optim

MIDDLE_SHAPE = 256

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

class ReplayDataset(dataset.Dataset):
    def __init__(self, replay_buffer):
        super().__init__()
        self.replay_buffer = replay_buffer
        self.size = len(replay_buffer.observations)
        print(self.size, len(replay_buffer.actions), len(replay_buffer.rewards))
        # self.observation_shape = replay_buffer.observations.shape[1:]
        # self.action_shape = replay_buffer.actions.shape[1:]

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        assert 0 <= item < self.size
        state = self.replay_buffer.observations[item]
        if item == 0:
            past_item = 0
        elif self.replay_buffer.dones_float[item - 1]:
            past_item = item
        else:
            past_item = item - 1
        past_state = self.replay_buffer.observations[past_item]
        action = self.replay_buffer.actions[item]
        reward = self.replay_buffer.rewards[item]
        # return torch.tensor(np.concatenate([state, past_state])), torch.tensor(action), torch.tensor(reward)
        return torch.tensor(state), torch.tensor(past_state), torch.tensor(action), torch.tensor(reward)


class RandomRewardEnsemble(nn.Module):
    def __init__(self, state_dim, action_dim, lr=1e-3, ensemble=1):
        super().__init__()
        self.rewards = [RandomReward(state_dim, action_dim).to(device) for _ in range(ensemble)]
        self.optimizers = [optim.Adam(model.parameters(), lr=lr, weight_decay=1.) for model in self.rewards]

    def __call__(self, state, prev_state, action):
        out = []
        for model in self.rewards:
            out.append(model(state, prev_state, action))
        return torch.cat(out, dim=1)
        # return self.rewards[0](state, prev_state, action)

    def get_optimizer(self):
        return self.optimizers


class RandomReward(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim, MIDDLE_SHAPE)
        self.l2 = nn.Linear(MIDDLE_SHAPE, MIDDLE_SHAPE)
        # self.l2_add = nn.Linear(MIDDLE_SHAPE, MIDDLE_SHAPE)
        self.l3 = nn.Linear(MIDDLE_SHAPE, 1)

        self.al1 = nn.Linear(action_dim, MIDDLE_SHAPE)
        self.al2 = nn.Linear(MIDDLE_SHAPE, MIDDLE_SHAPE)
        # self.al2_add = nn.Linear(MIDDLE_SHAPE, MIDDLE_SHAPE)
        self.al3 = nn.Linear(MIDDLE_SHAPE, 1)

    def forward(self, state, prev_state, action):
        # sa = torch.cat([state, action], 1)
        state_diff = state
        action_square = action ** 2
        q1 = F.relu(self.l1(state_diff))
        q1 = F.relu(self.l2(q1))
        # q1 = F.relu(self.l2_add(q1))
        q1 = self.l3(q1)

        c1 = F.relu(self.al1(action_square))
        c2 = F.relu(self.al2(c1))
        # c2 = F.relu(self.al2_add(c2))
        c3 = self.al3(c2)
        return q1 + c3


def mse_loss(x, y):
    info = dict()
    if len(x.shape) > 1 and x.shape[1] > 1:
        # ensemble
        std = torch.std(x, dim=-1).mean()
        # x = x.mean(dim=-1)
        y = y.reshape(-1, 1).repeat(1, x.shape[1])
        # print(y.shape, x.shape)
        # x = x[:, 0]
        info["std"] = std
        loss = ((x - y) ** 2).sum(axis=1).mean()
        # loss = torch.mean((x[:, 0] - y) ** 2)
    else:
        x, y = x.reshape(-1), y.reshape(-1)
        loss = torch.mean((x - y) ** 2)

    return loss, info


# Testing Loop
def run_one_epoch(epoch, model, optimizer, data_loader, writer, total_steps, train=True):
    # model.eval()
    tb_label = "train" if train else "test"
    cum_loss = 0.0
    cum_steps = 0
    cum_info = {}
    for i, data in tqdm(enumerate(data_loader)):
        states, prev_state, actions, rewards = data
        states, prev_state, actions, rewards = states.to(device), prev_state.to(device), actions.to(device), rewards.to(device)

        outputs = model(states, prev_state, actions)
        # print(outputs.shape, rewards.shape)
        loss, info = mse_loss(outputs, rewards)
        if train:
            loss.backward()
            if isinstance(optimizer, list):
                for opt in optimizer:
                    opt.step()
                    opt.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()

        cum_loss += loss.item()
        cum_steps += 1
        total_steps += 1
        if (cum_steps == 100 and train) or i == (len(data_loader) - 1):  # print every 100 mini-batches
            print(f'[{tb_label}] [{epoch + 1}, {cum_steps :5d}] loss: {cum_loss / cum_steps:.4f}')
            writer.add_scalar(f"{tb_label}/reward_learning_loss", cum_loss / cum_steps, total_steps)
            cum_loss = 0
            cum_steps = 0
            for key, value in info.items():
                try:
                    writer.add_scalar(f"{tb_label}/" + key, value, total_steps)
                except ValueError:
                    pass
                except AttributeError:
                    pass

            writer.flush()
    return total_steps


def get_output(train_loader, data_loader, model, ensemble, variant):
    final_result = []
    train_data_rewards = []
    delta = None
    model.eval()

    for i, train_data in tqdm(enumerate(train_loader)):
        train_states, train_prev_state, train_actions, train_rewards = train_data
        train_data_rewards.append(train_rewards)
    train_data_rewards = np.concatenate(train_data_rewards)
    print(train_data_rewards.shape, '----123')

    for i, data in tqdm(enumerate(data_loader)):
        states, prev_state, actions, rewards = data
        states, prev_state, actions, rewards = states.to(device), prev_state.to(device), actions.to(device), rewards.to(device)
        outputs = model(states, prev_state, actions).detach().cpu().numpy()
        final_result.append(outputs)
    if ensemble == 1:
        final_result = np.concatenate(final_result).reshape(-1)
    else:
        final_result = np.concatenate(final_result).reshape(-1, ensemble)
        mean = final_result.mean(axis=-1)
        std = final_result.std(axis=-1)
        # mean - std:
        # final_result = np.maximum(mean - variant['weight_std'] * std, 0)
        # L_min:
        # final_result = final_result.min(axis=-1)
        # L_min - std: 
        # final_result = np.maximum(final_result.min(axis=-1) - variant['weight_std'] * std, 0)
        # L_min - k * std: 
        train_data_rewards = train_data_rewards * 10
        final_result = final_result * 10
        delta = np.mean(train_data_rewards) - np.mean(final_result.min(axis=-1))
        print(delta, np.mean(train_data_rewards))
        data_rewards_mean = np.mean(train_data_rewards)
        if delta >= 0:
            delta = 28.5 / data_rewards_mean * delta
        else:
            delta = -0.3 * delta
        print(delta, data_rewards_mean, '----123')
        final_result = np.maximum(final_result.min(axis=-1) - delta * std, 0)
        
        print('trained mean:', np.mean(mean), 'final mean: ', np.mean(final_result), 'delta: ', delta, '\n')
        print('data mean: ', np.mean(train_data_rewards), 'trained std: ', np.mean(std), '\n')
        print(final_result, '\n', "="*30)
        print('final_result: ', final_result.shape)
    return final_result


def train(replay_buffer, source_replay_buffer, state_dim, action_dim, batch_size, num_epochs, lr, writer, num_workers,
          ensemble=1, variant=None):
    dataset = ReplayDataset(replay_buffer)
    source_dataset = ReplayDataset(source_replay_buffer)
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(source_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers)
    print('ensemble_numer: ', ensemble, '\n', '='*30)
    model = RandomRewardEnsemble(state_dim, action_dim, lr, ensemble=ensemble)
    model = model.to(device)
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-1)
    optimizer = model.get_optimizer()
    total_steps = 0

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        model.train()
        total_steps = run_one_epoch(epoch, model, optimizer, train_loader, writer, total_steps, True)
        model.eval()
        run_one_epoch(epoch, model, optimizer, test_loader, writer, total_steps, False)
    # torch.save(model.state_dict(), save_path)
    print('Finished Training')
    return get_output(train_loader, test_loader, model, ensemble, variant)
