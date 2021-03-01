import json
import os.path as osp
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from gan_layers import SpectralNormalization
from td3_models import Actor, Critic, DDPG
from sac_models import StochasticActor, SAC
from samplers import Sampler

import logz
from utils import load_expert_trajectories
from utils import log_trajectory_statistics
from envs.envs import (ExpertInvertedPendulumEnv, AgentInvertedPendulumEnv, ExpertInvertedDoublePendulumEnv,
                       AgentInvertedDoublePendulumEnv, ReacherEasyEnv, TiltedReacherEasyEnv, ThreeReacherEasyEnv,
                       Tilted3ReacherEasyEnv, ExpertHalfCheetahEnv, LockedLegsHalfCheetahEnv, HopperEnv,
                       HopperFlexibleEnv)
from envs.manipulation_envs import PusherEnv, PusherHumanSimEnv, StrikerEnv, StrikerHumanSimEnv
from buffers import LearnerAgentReplayBuffer, DemonstrationsReplayBuffer
from gail_models import InvariantDiscriminator, DeterministicPreprocessor, GaussianPreprocessor
from disentangail_models import DisentanGAIL, MIEstimator


def run_experiment(exp_params, learner_params, discriminator_params):
    # Experiment parameters
    file_location = exp_params.get('expert_samples_location', 'expert_data')
    prior_file_location = exp_params.get('prior_samples_location', 'prior_data')
    env_name = exp_params.get('env_name', 'InvertedPendulum-v2')
    env_type = exp_params.get('env_type', 'expert')
    exp_name = exp_params.get('exp_name', '{}_{}'.format(env_name, env_type))
    exp_num = exp_params.get('exp_num', 0)
    epochs = exp_params.get('epochs', 100)
    test_runs_per_epoch = exp_params.get('test_runs_per_epoch', 10)
    steps_per_epoch = exp_params.get('steps_per_epoch', 1000)
    init_random_samples = exp_params.get('init_random_samples', 5000)
    training_starts = exp_params.get('training_starts', 0)
    episode_limit = exp_params.get('episode_limit', 200)
    return_threshold = exp_params.get('return_threshold', 1e4)
    return_agent_buffer = exp_params.get('return_agent_buffer', False)
    visualize_collected_observations = exp_params.get('visualize_collected_observations', False)
    save_weights_checkpoints = exp_params.get('save_weights_checkpoints', False)

    # Learner parameters
    l_type = learner_params.get('l_type', 'TD3')
    l_buffer_size = learner_params.get('l_buffer_size', 10000)
    l_exploration_noise = learner_params.get('l_exploration_noise', 0.2)
    l_learning_rate = learner_params.get('l_learning_rate', 1e-3)
    l_batch_size = learner_params.get('l_batch_size', 128)
    l_updates_per_step = learner_params.get('l_updates_per_step', 1)
    l_act_delay = learner_params.get('l_act_delay', 2)
    l_gamma = learner_params.get('l_gamma', 0.99)
    l_polyak = learner_params.get('l_polyak', 0.995)
    l_train_actor_noise = learner_params.get('l_train_actor_noise', 0.1)
    l_entropy_coefficient = learner_params.get('l_entropy_coefficient', 0.2)
    l_tune_entropy_coefficient = learner_params.get('l_tune_entropy_coefficient', True)
    l_target_entropy = learner_params.get('l_target_entropy', None)
    l_clip_actor_gradients = learner_params.get('l_clip_actor_gradients', False)

    # Discriminator parameters
    d_type = discriminator_params.get('d_type', 'latent')
    d_loss = discriminator_params.get('d_loss', 'ce')
    d_rew = discriminator_params.get('d_rew', 'mixed')
    d_rew_noise = discriminator_params.get('d_rew_noise', True)
    d_learning_rate = discriminator_params.get('d_learning_rate', 1e-3)
    d_mi_learning_rate = discriminator_params.get('d_mi_learning_rate', 1e-3)
    d_updates_per_step = discriminator_params.get('d_updates_per_step', 1)
    d_mi_updates_per_step = discriminator_params.get('d_mi_updates_per_step', 1)
    d_e_batch_size = discriminator_params.get('d_e_batch_size', 64)
    d_l_batch_size = discriminator_params.get('d_l_batch_size', 64)
    d_label_smoothing = discriminator_params.get('d_label_smoothing', 0.0)
    d_stability_constant = discriminator_params.get('d_stability_constant', 0.0)
    d_sn_discriminator = discriminator_params.get('d_sn_discriminator', False)
    d_mi_constant = discriminator_params.get('d_mi_constant', 0.0)
    d_adaptive_mi = discriminator_params.get('d_adaptive_mi', False)
    d_double_mi = discriminator_params.get('d_double_mi', False)
    d_use_min_double_mi = discriminator_params.get('d_use_min_double_mi', False)
    d_max_mi = discriminator_params.get('d_max_mi', 1)
    d_min_mi = discriminator_params.get('d_min_mi', d_max_mi / 2)
    d_use_dual_mi = discriminator_params.get('d_use_dual_mi', False)
    d_mi_lagrangian_lr = discriminator_params.get('d_mi_lagrangian_lr', 1e-3)
    d_max_mi_constant = discriminator_params.get('d_max_mi_constant', 10)
    d_min_mi_constant = discriminator_params.get('d_min_mi_constant', 1e-4)
    d_unbiased_mi = discriminator_params.get('d_unbiased_mi', False)
    d_unbiased_mi_decay = discriminator_params.get('d_unbiased_mi_decay', 0.99)
    d_prior_mi_constant = discriminator_params.get('d_prior_mi_constant', 0.0)
    d_negative_priors = discriminator_params.get('d_negative_priors', False)
    d_max_mi_prior = discriminator_params.get('d_max_mi_prior', 0.05)
    d_min_mi_prior_constant = discriminator_params.get('d_min_mi_prior_constant', 1e-4)
    d_clip_mi_predictions = discriminator_params.get('d_clip_mi_predictions', False)
    d_pre_filters = discriminator_params.get('d_pre_filters', [32, 32, 1])
    d_hidden_units = discriminator_params.get('d_hidden_units', [32])
    d_mi_hidden_units = discriminator_params.get('d_mi_hidden_units', [32, 32])
    d_mi2_hidden_units = discriminator_params.get('d_mi2_hidden_units', d_mi_hidden_units)
    d_pre_scale_stddev = discriminator_params.get('d_pre_scale_stddev', 1.0)
    n_expert_demos = discriminator_params.get('n_expert_demos', None)
    n_expert_prior_demos = discriminator_params.get('n_expert_prior_demos', None)
    n_agent_prior_demos = discriminator_params.get('n_agent_prior_demos', n_expert_prior_demos)

    if env_name == 'InvertedPendulum-v2':
        im_side = 32
        im_shape = [im_side, im_side]
        expert_prior_location = 'Expert' + env_name
        if env_type == 'expert':
            env = ExpertInvertedPendulumEnv()
            agent_prior_location = 'Expert' + env_name
        elif env_type == 'agent' or env_type == 'colored' or env_type == 'to_colored':
            env = AgentInvertedPendulumEnv()
            agent_prior_location = 'Agent' + env_name
        elif env_type == 'to_two':
            env = ExpertInvertedDoublePendulumEnv()
            agent_prior_location = 'ExpertInvertedDoublePendulum-v2'
        elif env_type == 'to_colored_two':
            env = AgentInvertedDoublePendulumEnv()
            agent_prior_location = 'AgentInvertedDoublePendulum-v2'
        else:
            raise NotImplementedError
    elif env_name == 'InvertedDoublePendulum-v2':
        im_side = 32
        im_shape = [im_side, im_side]
        expert_prior_location = 'ExpertInvertedDoublePendulum-v2'
        if env_type == 'expert':
            agent_prior_location = 'ExpertInvertedDoublePendulum-v2'
            env = ExpertInvertedDoublePendulumEnv()
        elif env_type == 'colored' or env_type == 'to_colored':
            env = AgentInvertedDoublePendulumEnv()
            agent_prior_location = 'AgentInvertedDoublePendulum-v2'
        elif env_type == 'to_one':
            agent_prior_location = 'ExpertInvertedPendulum-v2'
            env = ExpertInvertedPendulumEnv()
        elif env_type == 'agent' or env_type == 'to_colored_one':
            agent_prior_location = 'AgentInvertedPendulum-v2'
            env = AgentInvertedPendulumEnv()
        else:
            raise NotImplementedError
    elif env_name == 'ThreeReacherEasy-v2':
        im_side = 48
        im_shape = [im_side, im_side]
        expert_prior_location = 'Expert' + env_name
        if env_type == 'expert':
            env = ThreeReacherEasyEnv()
            agent_prior_location = 'Expert' + env_name
        elif env_type == 'agent' or env_type == 'to_two':
            agent_prior_location = 'ExpertReacherEasy-v2'
            env = ReacherEasyEnv()
        elif env_type == 'tilted' or env_type == 'to_tilted':
            agent_prior_location = 'AgentThreeReacherEasy-v2'
            env = Tilted3ReacherEasyEnv()
        elif env_type == 'to_tilted_two':
            env = TiltedReacherEasyEnv()
            agent_prior_location = 'AgentReacherEasy-v2'
        else:
            raise NotImplementedError
    elif env_name == 'ReacherEasy-v2':
        im_side = 48
        im_shape = [im_side, im_side]
        expert_prior_location = 'ExpertReacherEasy-v2'
        if env_type == 'expert':
            env = ReacherEasyEnv()
            agent_prior_location = 'ExpertReacherEasy-v2'
        elif env_type == 'agent' or env_type == 'tilted' or env_type == 'to_tilted':
            env = TiltedReacherEasyEnv()
            agent_prior_location = 'AgentReacherEasy-v2'
        elif env_type == 'to_three':
            env = ThreeReacherEasyEnv()
            agent_prior_location = 'ExpertThreeReacherEasy-v2'
        elif env_type == 'to_tilted_three':
            agent_prior_location = 'AgentThreeReacherEasy-v2'
            env = Tilted3ReacherEasyEnv()
        else:
            raise NotImplementedError
    elif env_name == 'Hopper-v2':
        im_side = 64
        im_shape = [im_side, im_side]
        expert_prior_location = 'Hopper-v2'
        if env_type == 'expert':
            env = HopperEnv()
            agent_prior_location = 'Hopper-v2'
        elif env_type == 'flexible':
            env = HopperFlexibleEnv()
            agent_prior_location = 'HopperFlexible-v2'
        else:
            raise NotImplementedError
    elif env_name == 'HalfCheetah-v2':
        im_side = 64
        im_shape = [im_side, im_side]
        expert_prior_location = 'HalfCheetah-v2'
        if env_type == 'expert':
            env = ExpertHalfCheetahEnv()
            agent_prior_location = 'HalfCheetah-v2'
        elif env_type == 'locked_legs':
            env = LockedLegsHalfCheetahEnv()
            agent_prior_location = 'LockedLegsHalfCheetah-v2'
        else:
            raise NotImplementedError
    elif env_name == 'Striker-v2':
        im_side = 48
        im_shape = [im_side, im_side]
        expert_prior_location = 'Striker-v2'
        if env_type == 'expert':
            env = StrikerEnv()
            agent_prior_location = 'Striker-v2'
        elif env_type == 'to_human':
            env = StrikerHumanSimEnv()
            agent_prior_location = 'StrikerHuman-v2'
        else:
            raise NotImplementedError
    elif env_name == 'StrikerHumanSim-v2':
        im_side = 48
        im_shape = [im_side, im_side]
        expert_prior_location = 'StrikerHumanSim-v2'
        if env_type == 'expert':
            env = StrikerHumanSimEnv()
            agent_prior_location = 'StrikerHumanSim-v2'
        elif env_type == 'to_robot':
            env = StrikerEnv()
            agent_prior_location = 'Striker-v2'
        else:
            raise NotImplementedError
    elif env_name == 'Pusher-v2':
        im_side = 48
        im_shape = [im_side, im_side]
        expert_prior_location = 'Pusher-v2'
        if env_type == 'expert':
            env = PusherEnv()
            agent_prior_location = 'Pusher-v2'
        elif env_type == 'to_human':
            env = PusherHumanSimEnv()
            agent_prior_location = 'PusherHuman-v2'
        else:
            raise NotImplementedError
    elif env_name == 'PusherHumanSim-v2':
        im_side = 48
        im_shape = [im_side, im_side]
        expert_prior_location = 'PusherHumanSim-v2'
        if env_type == 'expert':
            env = PusherHumanSimEnv()
            agent_prior_location = 'PusherHumanSim-v2'
        elif env_type == 'to_robot':
            env = PusherEnv()
            agent_prior_location = 'Pusher-v2'
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    expert_buffer = DemonstrationsReplayBuffer(
        load_expert_trajectories(env_name, file_location, visual_data=True, load_ids=True,
                                 max_demos=n_expert_demos))
    expert_visual_data_shape = expert_buffer.get_random_batch(1)['ims'][0].shape
    print('Visual data shape: {}'.format(expert_visual_data_shape))
    past_frames = expert_visual_data_shape[0]
    print('Past frames: {}'.format(past_frames))
    if d_prior_mi_constant > 0.0 or d_negative_priors:
        prior_expert_buffer = DemonstrationsReplayBuffer(load_expert_trajectories(
            agent_prior_location, prior_file_location, visual_data=True, load_ids=True,
            max_demos=n_expert_prior_demos))
        prior_agent_buffer = DemonstrationsReplayBuffer(load_expert_trajectories(
            expert_prior_location, prior_file_location, visual_data=True, load_ids=True,
            max_demos=n_agent_prior_demos))
    else:
        prior_expert_buffer, prior_agent_buffer = None, None

    if d_type == 'latent':
        im_shape += [3]
    else:
        im_shape += [3 * past_frames]

    action_size = env.action_space.shape[0]
    if exp_num == -1:
        logz.configure_output_dir(None, True)
    else:
        log_dir = osp.join('experiments_data/', '{}/{}'.format(exp_name, exp_num))
        logz.configure_output_dir(log_dir, True)

    params = {
        'exp': exp_params,
        'learner': learner_params,
        'discriminator': discriminator_params,
    }
    print(params)
    logz.save_params(params)
    if l_type == 'TD3':
        def make_actor():
            actor = Actor([tf.keras.layers.Dense(400, 'relu', kernel_initializer='orthogonal'),
                           tf.keras.layers.Dense(300, 'relu', kernel_initializer='orthogonal'),
                           tf.keras.layers.Dense(action_size, 'tanh',
                                                 kernel_initializer=tf.keras.initializers.Orthogonal(0.01))])
            return actor

        def make_critic():
            critic = Critic([tf.keras.layers.Dense(400, 'relu', kernel_initializer='orthogonal'),
                             tf.keras.layers.Dense(300, 'relu', kernel_initializer='orthogonal'),
                             tf.keras.layers.Dense(1,
                                                   kernel_initializer=tf.keras.initializers.Orthogonal(0.01))])
            return critic
    elif l_type == 'SAC':
        def make_actor():
            actor = StochasticActor([tf.keras.layers.Dense(256, 'relu', kernel_initializer='orthogonal'),
                                     tf.keras.layers.Dense(256, 'relu', kernel_initializer='orthogonal'),
                                     tf.keras.layers.Dense(action_size * 2,
                                                           kernel_initializer=tf.keras.initializers.Orthogonal(0.01))])
            return actor

        def make_critic():
            critic = Critic([tf.keras.layers.Dense(256, 'relu', kernel_initializer='orthogonal'),
                             tf.keras.layers.Dense(256, 'relu', kernel_initializer='orthogonal'),
                             tf.keras.layers.Dense(1,
                                                   kernel_initializer=tf.keras.initializers.Orthogonal(0.01))])
            return critic

        if l_target_entropy is None:
            l_target_entropy = -1 * (np.prod(env.action_space.shape))
    else:
        raise NotImplementedError

    d_optimizer = tf.keras.optimizers.Adam(learning_rate=d_learning_rate)
    d_mi_optimizer = tf.keras.optimizers.Adam(learning_rate=d_mi_learning_rate)
    d_mi_lagrangian_optimizer = tf.keras.optimizers.Adam(learning_rate=d_mi_lagrangian_lr)
    tfl = tf.keras.layers
    if d_type == 'latent':
        pre_layers = [tfl.Reshape(im_shape)]
    else:
        pre_layers = [tfl.Permute((2, 3, 1, 4)),
                      tfl.Reshape(im_shape)]
    if (d_type == 'latent') or (not d_sn_discriminator):
        for filters in d_pre_filters[:-1]:
            pre_layers += [tfl.Conv2D(filters, 3, activation='tanh', padding='same'),
                           tfl.MaxPooling2D(2, padding='same')]
        pre_layers += [tfl.Conv2D(d_pre_filters[-1], 3, padding='same'),
                       tfl.MaxPooling2D(2, padding='same'),
                       tfl.Reshape([-1])]
    else:
        for filters in d_pre_filters[:-1]:
            pre_layers += [SpectralNormalization(
                tfl.Conv2D(filters, 3, padding='same')),
                tfl.LeakyReLU(),
                tfl.MaxPooling2D(2, padding='same')]
        pre_layers += [SpectralNormalization(
            tfl.Conv2D(d_pre_filters[-1], 3, padding='same')),
            tfl.MaxPooling2D(2, padding='same'),
            tfl.Reshape([-1])]
    if d_sn_discriminator:
        disc_layers = [SpectralNormalization(
            tfl.Dense(units, activation='relu'))
            for units in d_hidden_units]
        disc_layers.append(SpectralNormalization(tfl.Dense(1)))
    else:
        disc_layers = [tfl.Dense(units, activation='tanh')
                       for units in d_hidden_units]
        disc_layers.append(tfl.Dense(1))
    if d_type == 'latent':
        def make_pre():
            pre = GaussianPreprocessor(pre_layers, d_pre_scale_stddev)
            return pre

        def make_disc():
            disc = InvariantDiscriminator(disc_layers,
                                          d_stability_constant,
                                          d_rew)
            return disc
    else:
        def make_pre():
            pre = DeterministicPreprocessor(pre_layers)
            return pre

        def make_disc():
            disc = InvariantDiscriminator(disc_layers,
                                          d_stability_constant,
                                          d_rew)
            return disc
    mi_layers = [tfl.Dense(units, activation='tanh') for units in d_mi_hidden_units]
    mi_layers.append(tfl.Dense(1))

    def make_mi_est():
        mi_est = MIEstimator(mi_layers)
        return mi_est

    if d_double_mi:
        mi2_layers = [tfl.Dense(units, activation='tanh') for units in d_mi2_hidden_units]
        mi2_layers.append(tfl.Dense(1))

        def make_mi2_est():
            mi2_est = MIEstimator(mi2_layers)
            return mi2_est
    else:
        make_mi2_est = None

    l_optimizer = tf.keras.optimizers.Adam(l_learning_rate)
    if l_type == 'TD3':
        l_agent = DDPG(make_actor=make_actor,
                       make_critic=make_critic,
                       make_critic2=make_critic,
                       actor_optimizer=l_optimizer,
                       critic_optimizer=l_optimizer,
                       gamma=l_gamma,
                       polyak=l_polyak,
                       train_actor_noise=l_train_actor_noise,
                       clip_actor_gradients=l_clip_actor_gradients,)
    elif l_type == 'SAC':
        l_agent = SAC(make_actor=make_actor,
                      make_critic=make_critic,
                      make_critic2=make_critic,
                      actor_optimizer=l_optimizer,
                      critic_optimizer=l_optimizer,
                      gamma=l_gamma,
                      polyak=l_polyak,
                      entropy_coefficient=l_entropy_coefficient,
                      tune_entropy_coefficient=l_tune_entropy_coefficient,
                      target_entropy=l_target_entropy,
                      clip_actor_gradients=l_clip_actor_gradients,)
    else:
        raise NotImplementedError
    sampler = Sampler(env, episode_limit, init_random_samples, visual_env=True)

    gail = DisentanGAIL(agent=l_agent,
                        make_discriminator=make_disc,
                        make_preprocessing=make_pre,
                        expert_buffer=expert_buffer,
                        prior_expert_buffer=prior_expert_buffer,
                        prior_agent_buffer=prior_agent_buffer,
                        make_mi_estimator=make_mi_est,
                        make_mi2_estimator=make_mi2_est,
                        use_min_double_mi=d_use_min_double_mi,
                        d_loss=d_loss,
                        d_optimizer=d_optimizer,
                        mi_optimizer=d_mi_optimizer,
                        label_smoothing=d_label_smoothing,
                        stab_const=d_stability_constant,
                        mi_constant=d_mi_constant,
                        adaptive_mi=d_adaptive_mi,
                        max_mi=d_max_mi,
                        min_mi=d_min_mi,
                        prior_mi_constant=d_prior_mi_constant,
                        negative_priors=d_negative_priors,
                        max_mi_prior=d_max_mi_prior,
                        use_dual_mi=d_use_dual_mi,
                        mi_lagrangian_optimizer=d_mi_lagrangian_optimizer,
                        max_mi_constant=d_max_mi_constant,
                        min_mi_constant=d_min_mi_constant,
                        min_mi_prior_constant=d_min_mi_prior_constant,
                        unbiased_mi=d_unbiased_mi,
                        clip_mi_predictions=d_clip_mi_predictions,
                        unbiased_mi_decay=d_unbiased_mi_decay,
                        im_side=im_side,
                        past_frames=past_frames,)

    agent_buffer = LearnerAgentReplayBuffer(gail, l_buffer_size, reward_noise=d_rew_noise)
    test_input = expert_buffer.get_random_batch(1)
    test_input['obs'] = np.expand_dims(
        (env.reset()['obs']).astype('float32'), axis=0)
    gail(test_input)
    gail.summary()

    mean_test_returns = []
    mean_test_std = []
    steps = []

    step_counter = 0
    logz.log_tabular('Iteration', 0)
    logz.log_tabular('Steps', step_counter)
    print('Epoch {}/{} - total steps {}'.format(0, epochs, step_counter))
    out = sampler.evaluate(l_agent, test_runs_per_epoch, False)
    mean_test_returns.append(out['mean'])
    mean_test_std.append(out['std'])
    steps.append(step_counter)
    for k, v in out.items():
        logz.log_tabular(k, v)
    logz.dump_tabular()
    for e in range(epochs):
        while step_counter < (e + 1) * steps_per_epoch:
            traj_data = sampler.sample_trajectory(l_agent, l_exploration_noise)
            agent_buffer.add(traj_data)
            n = traj_data['n']
            step_counter += traj_data['n']
            if step_counter > training_starts:
                gail.train(agent_buffer=agent_buffer,
                           l_batch_size=l_batch_size,
                           l_updates=l_updates_per_step * n,
                           l_act_delay=l_act_delay,
                           d_updates=d_updates_per_step * n,
                           mi_updates=d_mi_updates_per_step * n,
                           d_e_batch_size=d_e_batch_size,
                           d_l_batch_size=d_l_batch_size,)

        logz.log_tabular('Iteration', e + 1)
        logz.log_tabular('Steps', step_counter)
        print('Epoch {}/{} - total steps {}'.format(e + 1, epochs, step_counter))
        traj_test = sampler.sample_test_trajectories(l_agent, 0.0, test_runs_per_epoch)
        out = log_trajectory_statistics(traj_test['ret'], False)
        mean_test_returns.append(out['mean'])
        mean_test_std.append(out['std'])
        steps.append(step_counter)
        for k, v in out.items():
            logz.log_tabular(k, v)
        logz.dump_tabular()
        if save_weights_checkpoints:
            weights_log_dir = 'experiments_data/{}/{}/{}/{}.h5'.format(exp_name, exp_num, 'weights', e)
            l_agent.save_weights(weights_log_dir)

        if visualize_collected_observations:
            training_sample = traj_data['ims'][-1, 0]
            print('Visualization of latest training sample')
            plt.imshow(training_sample)
            plt.show()
            test_sample = traj_test['ims'][-1, 0]
            print('Visualization of latest test sample')
            plt.imshow(test_sample)
            plt.show()
        if out['mean'] >= return_threshold:
            print('Early termination due to reaching return threshold')
            break

    if return_agent_buffer:
        return gail, sampler, agent_buffer
    else:
        return gail, sampler,


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run experiment using DisentanGAIL with given parameters file.')
    parser.add_argument('parameters', default='parameters/params_dgail.json', help='Parameters file location.')
    args = parser.parse_args()
    with open(args.parameters, 'r') as inp:
        parameters = json.loads(inp.read())
    run_experiment(exp_params=parameters['exp'],
                   learner_params=parameters['learner'],
                   discriminator_params=parameters['discriminator'])


if __name__ == '__main__':
    main()
