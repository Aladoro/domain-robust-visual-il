import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from sac_models import StochasticActor, Critic, SAC
from samplers import Sampler
from buffers import ReplayBuffer
from envs.envs import (ExpertInvertedPendulumEnv, AgentInvertedPendulumEnv, ExpertInvertedDoublePendulumEnv,
                       AgentInvertedDoublePendulumEnv, ReacherEasyEnv, TiltedReacherEasyEnv, ThreeReacherEasyEnv,
                       Tilted3ReacherEasyEnv, ExpertHalfCheetahEnv, LockedLegsHalfCheetahEnv, HopperEnv,
                       HopperFlexibleEnv)
from envs.manipulation_envs import PusherEnv, PusherHumanSimEnv, StrikerEnv, StrikerHumanSimEnv

def train_expert(env_name):
    """Train expert policy in given environment."""
    if env_name == 'InvertedPendulum-v2':
        env = ExpertInvertedPendulumEnv()
        episode_limit = 200
        return_threshold = 200
    elif env_name == 'InvertedDoublePendulum-v2':
        env = ExpertInvertedDoublePendulumEnv()
        episode_limit = 50
        return_threshold = 460
    elif env_name == 'ThreeReacherEasy-v2':
        env = ThreeReacherEasyEnv()
        episode_limit = 50
        return_threshold = -0.8
    elif env_name == 'ReacherEasy-v2':
        env = ReacherEasyEnv()
        episode_limit = 50
        return_threshold = -0.8
    elif env_name == 'Hopper-v2':
        env = HopperEnv()
        episode_limit = 200
        return_threshold = 600
    elif env_name == 'HalfCheetah-v2':
        env = ExpertHalfCheetahEnv()
        episode_limit = 200
        return_threshold = 1000
    elif env_name == 'StrikerHumanSim-v2':
        env = StrikerHumanSimEnv()
        episode_limit = 200
        return_threshold = -190
    elif env_name == 'PusherHumanSim-v2':
        env = PusherHumanSimEnv()
        episode_limit = 200
        return_threshold = -80
    else:
        raise NotImplementedError
    buffer_size = 1000000
    init_random_samples = 1000
    exploration_noise = 0.2
    learning_rate = 3e-4
    batch_size = 256
    epochs = 200
    steps_per_epoch = 5000
    updates_per_step = 1
    update_actor_every = 1
    start_training = 512
    gamma = 0.99
    polyak = 0.995
    entropy_coefficient = 0.2
    clip_actor_gradients = False
    visual_env = True
    action_size = env.action_space.shape[0]
    tune_entropy_coefficient = True
    target_entropy = -1*action_size

    def make_actor():
      actor = StochasticActor([tf.keras.layers.Dense(256, 'relu'),
                    tf.keras.layers.Dense(256, 'relu'),
                    tf.keras.layers.Dense(action_size*2)])
      return actor

    def make_critic():
      critic = Critic([tf.keras.layers.Dense(256, 'relu'),
                    tf.keras.layers.Dense(256, 'relu'),
                    tf.keras.layers.Dense(1)])
      return critic
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    replay_buffer = ReplayBuffer(buffer_size)
    sampler = Sampler(env, episode_limit=episode_limit,
                      init_random_samples=init_random_samples, visual_env=visual_env)
    agent = SAC(make_actor,
                make_critic,
                make_critic,
                actor_optimizer=optimizer,
                critic_optimizer=optimizer,
                gamma=gamma,
                polyak=polyak,
                entropy_coefficient=entropy_coefficient,
                tune_entropy_coefficient=tune_entropy_coefficient,
                target_entropy=target_entropy,
                clip_actor_gradients=clip_actor_gradients)
    if visual_env:
        obs = np.expand_dims(env.reset()['obs'], axis=0)
    else:
        obs = np.expand_dims(env.reset(), axis=0)
    agent(obs)
    agent.summary()

    mean_test_returns = []
    mean_test_std = []
    steps = []

    step_counter = 0
    for e in range(epochs):
        while step_counter < (e + 1) * steps_per_epoch:
            traj_data = sampler.sample_trajectory(agent, exploration_noise)
            replay_buffer.add(traj_data)
            if step_counter > start_training:
                agent.train(replay_buffer, batch_size=batch_size,
                            n_updates=updates_per_step * traj_data['n'],
                            act_delay=update_actor_every)
            step_counter += traj_data['n']
        print('Epoch {}/{} - total steps {}'.format(e + 1, epochs, step_counter))
        out = sampler.evaluate(agent, 10)
        mean_test_returns.append(out['mean'])
        mean_test_std.append(out['std'])
        steps.append(step_counter)
        if out['mean'] >= return_threshold:
            print('Early termination due to reaching return threshold')
            break
    plt.errorbar(steps, mean_test_returns, mean_test_std)
    plt.xlabel('steps')
    plt.ylabel('returns')
    plt.show()
    return agent
