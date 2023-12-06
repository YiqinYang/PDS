import os
from typing import Tuple

import gym
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

import wrappers
from dataset_utils import D4RLDataset, split_into_trajectories
from evaluation import evaluate
from learner import Learner
import time

import preprocessing
from preprocessing import learned_reward
from preprocessing.learned_reward import merge_dataset

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'halfcheetah-expert-v2', 'Environment name.')
flags.DEFINE_string('source_name', 'halfcheetah-random-v2', 'Source Environment name.')
flags.DEFINE_string('comment', 'data_sharing', 'Comment for the run')

flags.DEFINE_string('save_dir', './runs/', 'Tensorboard logging dir.')
flags.DEFINE_string('data_share', 'none', 'share type.')
flags.DEFINE_float('target_split', 1, 'amount of target data to use')
flags.DEFINE_float('source_split', 1, 'amount of target data to use')
flags.DEFINE_float('weight', 10, 'std weight')
flags.DEFINE_float('ensemble', 10, 'ensemble')
# flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('seed', int(time.time()), 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
config_flags.DEFINE_config_file(
    'config',
    'default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def normalize(dataset, split=1.):
    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0

    if split < 1:
        assert split > 0
        sample_index = np.random.choice(np.arange(dataset.size), int(dataset.size * split), replace=False)
        dataset.rewards = dataset.rewards[sample_index, ...]
        dataset.observations = dataset.observations[sample_index, ...]
        dataset.next_observations = dataset.next_observations[sample_index, ...]
        dataset.actions = dataset.actions[sample_index, ...]
        dataset.dones_float = dataset.dones_float[sample_index, ...]
        dataset.masks = dataset.masks[sample_index, ...]
        dataset.size = int(dataset.size * split)


def make_env_and_dataset(env_name: str, target_split: int, source_env_name: str, source_split: int,
                         seed: int, writer: any, variant) -> Tuple[gym.Env, D4RLDataset]:
    env = gym.make(env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    source_env = gym.make(source_env_name)
    source_dataset = D4RLDataset(source_env)
    dataset = D4RLDataset(env)

    if 'antmaze' in FLAGS.env_name:
        dataset.rewards -= 1.0
        source_dataset.rewards -= 1.0
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        # but I found no difference between (x - 0.5) * 4 and x - 1.0
    elif ('halfcheetah' in FLAGS.env_name or 'walker2d' in FLAGS.env_name
          or 'hopper' in FLAGS.env_name):
        normalize(dataset, target_split)
        normalize(source_dataset, source_split)

    print('target dataset: ', dataset.rewards.shape, 'other data: ', source_dataset.rewards.shape, '\n', '='*30)
    merge_dataset(dataset, source_dataset, FLAGS.data_share, writer, variant)

    return env, dataset


def main(_):
    hidden_dim = preprocessing.training_reward.MIDDLE_SHAPE
    variant = {'weight_std': FLAGS.weight, 'hidden_dim': hidden_dim}
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # hidden_add:
    # summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir,
    # f"{str(variant['weight_std'])}_hidden_add_{str(variant['hidden_dim'])}_{FLAGS.data_share}_{FLAGS.env_name}_{FLAGS.source_name}_{str(FLAGS.seed)}"),
    #                                write_to_disk=True)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # learn_ensemble:
    # summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir,
    # f"{str(variant['weight_std'])}_{FLAGS.data_share}_{FLAGS.env_name}_{FLAGS.source_name}_{str(FLAGS.seed)}"),
    #                                write_to_disk=True)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # early_stopping:
    # train_epoch = learned_reward.NUM_EPOCHS
    # summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir,
    # f"early_stop_{str(variant['weight_std'])}_train_epoch_{str(train_epoch)}_{FLAGS.data_share}_{FLAGS.env_name}_{FLAGS.source_name}_{str(FLAGS.seed)}"),
    #                                write_to_disk=True)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # L_min_std:
    # summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir,
    # f"L_min_ensemble_{str(variant['weight_std'])}_{FLAGS.data_share}_{FLAGS.env_name}_{FLAGS.source_name}_{str(FLAGS.seed)}"),
    #                                write_to_disk=True)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # L_min source splits
    # learned_reward.ENSEMBLE = int(FLAGS.ensemble)
    # summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir,
    # f"L_min_ensemble_{str(FLAGS.ensemble)}_{FLAGS.data_share}_{FLAGS.env_name}_{FLAGS.source_name}_source_split_{str(FLAGS.source_split)}_{str(FLAGS.seed)}"),
    #                                write_to_disk=True)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # L_min delta
    summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir,
    f"L_min_delta_{FLAGS.data_share}_{FLAGS.env_name}_{FLAGS.source_name}_source_split_{str(FLAGS.source_split)}_{str(FLAGS.seed)}"),
                                   write_to_disk=True)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir,
    # f"{str(variant['weight_std'])}_{FLAGS.data_share}_{FLAGS.env_name}_{FLAGS.source_name}_{str(FLAGS.seed)}"),
    #                                write_to_disk=True)

    os.makedirs(FLAGS.save_dir, exist_ok=True)

    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.target_split, FLAGS.source_name, FLAGS.source_split,
                                        FLAGS.seed, summary_writer, variant)
    
    kwargs = dict(FLAGS.config)
    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    max_steps=FLAGS.max_steps,
                    **kwargs)

    eval_returns = []
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        batch = dataset.sample(FLAGS.batch_size)

        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f'training/{k}', v, i)
                else:
                    summary_writer.add_histogram(f'training/{k}', v, i)
            summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)
            summary_writer.flush()

            eval_returns.append((i, eval_stats['return']))


if __name__ == '__main__':
    app.run(main)
