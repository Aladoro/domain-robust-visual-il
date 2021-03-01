import os

from utils import log_trajectory_statistics
from envs.envs import (ExpertInvertedPendulumEnv, AgentInvertedPendulumEnv, ExpertInvertedDoublePendulumEnv,
                       AgentInvertedDoublePendulumEnv, ReacherEasyEnv, TiltedReacherEasyEnv, ThreeReacherEasyEnv,
                       Tilted3ReacherEasyEnv, ExpertHalfCheetahEnv, LockedLegsHalfCheetahEnv, HopperEnv,
                       HopperFlexibleEnv)
from envs.manipulation_envs import PusherEnv, PusherHumanSimEnv, StrikerEnv, StrikerHumanSimEnv
from samplers import Sampler
from utils import save_expert_trajectories


def collect_expert_data(agent, env_name, max_timesteps=40000, expert_samples_location='expert_data'):
    """Collect and save demonstrations with trained expert agent.

    Parameters
    ----------
    agent : Trained expert agent.
    env_name : Source environment to collect the demonstrations.
    max_timesteps : Maximum number of visual observations to collect, default is 40000.
    expert_samples_location : Folder to save the expert demonstrations collected.
    """
    if env_name == 'InvertedPendulum-v2':
        expert_env = ExpertInvertedPendulumEnv()
        episode_limit = 200
    elif env_name == 'InvertedDoublePendulum-v2':
        expert_env = ExpertInvertedDoublePendulumEnv()
        episode_limit = 50
    elif env_name == 'ThreeReacherEasy-v2':
        expert_env = ThreeReacherEasyEnv()
        episode_limit = 50
    elif env_name == 'ReacherEasy-v2':
        expert_env = ReacherEasyEnv()
        episode_limit = 50
    elif env_name == 'Hopper-v2':
        expert_env = HopperEnv()
        episode_limit = 200
    elif env_name == 'HalfCheetah-v2':
        expert_env = ExpertHalfCheetahEnv()
        episode_limit = 200
    elif env_name == 'PusherHumanSim-v2':
        expert_env = PusherHumanSimEnv()
        episode_limit = 200
    elif env_name == 'StrikerHumanSim-v2':
        expert_env = StrikerHumanSimEnv()
        episode_limit = 200
    else:
        print('Please select one of the implemented environments:'
              '(InvertedPendulum-v2, InvertedDoublePendulum-v2, ReacherEasy-v2,'
              'ThreeReacherEasy-v2, Hopper-v2, HalfCheetah-v2, PusherHumanSim-v2,'
              'StrikerHumanSim-v2)')
        raise NotImplementedError
    episodes_n = int(max_timesteps // episode_limit)

    saver_sampler = Sampler(expert_env, episode_limit=episode_limit,
                            init_random_samples=0, visual_env=True)
    traj = saver_sampler.sample_test_trajectories(agent, 0.0,
                                                  episodes_n, False)
    log_trajectory_statistics(traj['ret'])
    os.makedirs(expert_samples_location + '/' + env_name)
    save_expert_trajectories(traj, env_name, expert_samples_location,
                             visual_data=True)
    print('Expert trajectories successfully saved.')
