import numpy as np
from preprocessing.training_reward import train
NUM_EPOCHS = 3
ENSEMBLE = 10


def reward_learning(dataset, source_dataset, state_dim, action_dim, writer, ensemble=1, variant=None):
    batch_size = 256
    lr = 1e-3
    num_epochs = NUM_EPOCHS
    num_workers = 8
    learned_rewards = train(dataset, source_dataset, state_dim, action_dim, batch_size, num_epochs, lr, writer,
                            num_workers, ensemble, variant)
    return learned_rewards


def merge_dataset(dataset, source_dataset, strategy="none", writer=None, variant=None):
    print(dataset.observations.shape, dataset.next_observations.shape, 
            dataset.actions.shape, dataset.dones_float.shape, dataset.masks.shape, 
                dataset.rewards.shape, dataset.size)

    print(source_dataset.observations.shape, source_dataset.next_observations.shape, 
            source_dataset.actions.shape, source_dataset.dones_float.shape, source_dataset.masks.shape, 
                source_dataset.rewards.shape, source_dataset.size)

    if strategy == "none":
        # don't share data
        return
    if strategy == "all":
        # share oracle rewards
        dataset.rewards = np.concatenate([dataset.rewards, source_dataset.rewards])
    elif strategy == "learn":
        state_dim = dataset.observations.shape[1]
        action_dim = dataset.actions.shape[1]
        learned_rewards = reward_learning(dataset, source_dataset, state_dim, action_dim, writer)
        dataset.rewards = np.concatenate([dataset.rewards, learned_rewards])
    elif strategy == "pess":
        state_dim = dataset.observations.shape[1]
        action_dim = dataset.actions.shape[1]
        learned_rewards = reward_learning(dataset, source_dataset, state_dim, action_dim, writer, ensemble=ENSEMBLE, variant=variant)
        dataset.rewards = np.concatenate([dataset.rewards, learned_rewards])
    elif strategy == "zero":
        # uds
        zero_rewards = np.zeros_like(source_dataset.rewards)
        dataset.rewards = np.concatenate([dataset.rewards, zero_rewards])
    else:
        print(f"Strategy {strategy} not found")
        raise NotImplementedError

    dataset.observations = np.concatenate([dataset.observations, source_dataset.observations])
    dataset.next_observations = np.concatenate([dataset.next_observations, source_dataset.next_observations])
    dataset.actions = np.concatenate([dataset.actions, source_dataset.actions])
    dataset.dones_float = np.concatenate([dataset.dones_float, source_dataset.dones_float])
    dataset.masks = np.concatenate([dataset.masks, source_dataset.masks])
    dataset.size = dataset.size + source_dataset.size

    saved_data = {'observations': dataset.observations,
                    'next_observations': dataset.next_observations,
                    'actions': dataset.actions,
                    'rewards': dataset.rewards,
                    'dones':dataset.dones_float}
    saved_name = variant['envname'] + '_' + variant['sourcename'] + '_' + str(variant['sourcesplit'])
    # saved_name = 'learn'+ '_' + variant['envname'] + '_' + variant['sourcename'] + '_' + str(variant['sourcesplit'])
    np.save(saved_name, saved_data)
    print('save data done')
    # print(dataset.rewards.shape, dataset.observations.shape, dataset.next_observations.shape, dataset.actions.shape, 
    #     dataset.dones_float.shape , '---123')
