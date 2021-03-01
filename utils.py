import numpy as np


def log_trajectory_statistics(trajectory_rewards, log=True):
    """Log and return trajectory statistics."""
    out = {}
    out['n'] = len(trajectory_rewards)
    out['mean'] = np.mean(trajectory_rewards)
    out['max'] = np.max(trajectory_rewards)
    out['min'] = np.min(trajectory_rewards)
    out['std'] = np.std(trajectory_rewards)
    if log:
        print('Number of completed trajectories - {}'.format(out['n']))
        print('Latest trajectories mean reward - {}'.format(out['mean']))
        print('Latest trajectories max reward - {}'.format(out['max']))
        print('Latest trajectories min reward - {}'.format(out['min']))
        print('Latest trajectories std reward - {}'.format(out['std']))
    return out


def save_expert_trajectories(trajectories, env_name, file_location, visual_data=False):
    """Save full visual trajectories data."""
    np.save(file_location + '/' + env_name + '/expert_obs.npy', trajectories['obs'])
    np.save(file_location + '/' + env_name + '/expert_acs.npy', trajectories['act'])
    np.save(file_location + '/' + env_name + '/expert_nobs.npy', trajectories['nobs'])
    np.save(file_location + '/' + env_name + '/expert_don.npy', trajectories['don'])
    np.save(file_location + '/' + env_name + '/expert_ids.npy', trajectories['ids'])
    if visual_data:
        np.save(file_location + '/' + env_name + '/expert_ims.npy', trajectories['ims'])


def load_expert_trajectories(env_name, file_location, visual_data=False, load_ids=False, max_demos=None):
    """Load full visual trajectories data."""
    if max_demos is None:
        out = {'obs': np.load(file_location + '/' + env_name + '/expert_obs.npy'),
               'act': np.load(file_location + '/' + env_name + '/expert_acs.npy'),
               'nobs': np.load(file_location + '/' + env_name + '/expert_nobs.npy'),
               'don': np.load(file_location + '/' + env_name + '/expert_don.npy')}
        if visual_data:
            out['ims'] = np.load(file_location + '/' + env_name + '/expert_ims.npy')
        if load_ids:
            out['ids'] = np.load(file_location + '/' + env_name + '/expert_ids.npy')
    else:
        out = {'obs': np.load(file_location + '/' + env_name + '/expert_obs.npy')[:max_demos],
               'act': np.load(file_location + '/' + env_name + '/expert_acs.npy')[:max_demos],
               'nobs': np.load(file_location + '/' + env_name + '/expert_nobs.npy')[:max_demos],
               'don': np.load(file_location + '/' + env_name + '/expert_don.npy')[:max_demos]}
        if visual_data:
            out['ims'] = np.load(file_location + '/' + env_name + '/expert_ims.npy')[:max_demos]
        if load_ids:
            out['ids'] = np.load(file_location + '/' + env_name + '/expert_ids.npy')[:max_demos]
    return out
