import os

from utils import log_trajectory_statistics
from envs.envs import (ExpertInvertedPendulumEnv, AgentInvertedPendulumEnv, ExpertInvertedDoublePendulumEnv,
                       AgentInvertedDoublePendulumEnv, ReacherEasyEnv, TiltedReacherEasyEnv, ThreeReacherEasyEnv,
                       Tilted3ReacherEasyEnv, ExpertHalfCheetahEnv, LockedLegsHalfCheetahEnv, HopperEnv,
                       HopperFlexibleEnv)
from envs.manipulation_envs import PusherEnv, PusherHumanSimEnv, StrikerEnv, StrikerHumanSimEnv
from samplers import Sampler
from utils import save_expert_trajectories


def collect_prior_data(realm_name, max_timesteps=40000, prior_samples_location='prior_data'):
    """Collect and save prior visual observations for an environment realm.

    Parameters
    ----------
    realm_name : Environment realm to collect the visual observations.
    max_timesteps : Maximum number of visual observations to collect, default is 40000.
    prior_samples_location : Folder to save the prior visual observations collected.
    """
    if realm_name == 'InvertedPendulum' or realm_name == 'Inverted Pendulum':
        prior_envs = [ExpertInvertedPendulumEnv(), AgentInvertedPendulumEnv(),
                      ExpertInvertedDoublePendulumEnv(), AgentInvertedDoublePendulumEnv()]
        prior_env_names = ['ExpertInvertedPendulum-v2', 'AgentInvertedPendulum-v2',
                           'ExpertInvertedDoublePendulum-v2', 'AgentInvertedDoublePendulum-v2']
        episode_limit = 50
    elif realm_name == 'Reacher':
        prior_envs = [ReacherEasyEnv(), TiltedReacherEasyEnv(),
                      ThreeReacherEasyEnv(), Tilted3ReacherEasyEnv()]
        prior_env_names = ['ExpertReacherEasy-v2', 'AgentReacherEasy-v2',
                           'ExpertThreeReacherEasy-v2', 'AgentThreeReacherEasy-v2']
        episode_limit = 50
    elif realm_name == 'Hopper':
        prior_envs = [HopperEnv(), HopperFlexibleEnv()]
        prior_env_names = ['Hopper-v2', 'HopperFlexible-v2']
        episode_limit = 200
    elif realm_name == 'HalfCheetah' or realm_name == 'Half-Cheetah':
        prior_envs = [ExpertHalfCheetahEnv(), LockedLegsHalfCheetahEnv()]
        prior_env_names = ['HalfCheetah-v2', 'LockedLegsHalfCheetah-v2']
        episode_limit = 200
    elif realm_name == 'Pusher' or realm_name == '7DOF-Pusher':
        prior_envs = [PusherEnv(), PusherHumanSimEnv()]
        prior_env_names = ['Pusher-v2', 'PusherHumanSim-v2']
        episode_limit = 200
    elif realm_name == 'Striker' or realm_name == '7DOF-Striker':
        prior_envs = [StrikerEnv(), StrikerHumanSimEnv()]
        prior_env_names = ['Striker-v2', 'StrikerHumanSim-v2']
        episode_limit = 200
    else:
        print('Please select one of the implemented realms:'
              '(InvertedPendulum/Inverted Pendulum, Reacher, '
              'Hopper, HalfCheetah/Half-Cheetah, '
              'Striker/7DOF-Striker, Pusher/7DOF-Pusher')
        raise NotImplementedError

    episodes_n = int(max_timesteps // episode_limit)

    for env, env_name in zip(prior_envs, prior_env_names):
        saver_sampler = Sampler(env, episode_limit=episode_limit,
                                init_random_samples=0, visual_env=True)
        traj = saver_sampler.sample_test_trajectories(None, 0.0, episodes_n, False)
        log_trajectory_statistics(traj['ret'])
        os.makedirs(prior_samples_location + '/' + env_name, exist_ok=True)
        save_expert_trajectories(traj, env_name, prior_samples_location,
                                 visual_data=True)
    print('Prior trajectories successfully saved.')
