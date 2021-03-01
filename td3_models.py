import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class Actor(tf.keras.layers.Layer):
    """Deterministic policy model."""
    def __init__(self, layers, norm_mean=None, norm_stddev=None):
        super(Actor, self).__init__()
        self._act_layers = layers
        self._out_dim = layers[-1].units
        self._norm_mean = norm_mean
        self._norm_stddev = norm_stddev

    def call(self, inputs):
        """Run network layers on input observations."""
        out = inputs
        for layer in self._act_layers:
            out = layer(out)
        return out

    def get_action(self, observation_batch, noise_stddev, max_noise=0.5, **kwargs):
        """Compute a single action sample from observations."""
        batch_dim = tf.shape(observation_batch)[0]
        pre_obs_batch = self._preprocess_obs(observation_batch)
        noise = tf.clip_by_value(tf.random.normal(shape=[batch_dim, self._out_dim],
                                                  stddev=noise_stddev), -max_noise, max_noise)
        return tf.clip_by_value(self.__call__(pre_obs_batch) + noise, -1, 1)

    def _preprocess_obs(self, observation_batch):
        if self._norm_mean is not None and self._norm_stddev is not None:
            return (observation_batch - self.norm_mean) / (self.norm_stddev + 1e-7)
        return observation_batch


class Critic(tf.keras.layers.Layer):
    """Simple Q-function model."""
    def __init__(self, layers, norm_mean=None, norm_stddev=None):
        super(Critic, self).__init__()
        self._cri_layers = layers
        self._norm_mean = norm_mean
        self._norm_stddev = norm_stddev

    def call(self, inputs):
        """Run network layers on inputs."""
        out = inputs
        for layer in self._cri_layers:
            out = layer(out)
        return out

    def get_q(self, observation_batch, action_batch):
        """Calculate Q values estimate from observations and actions."""
        pre_obs_batch = self._preprocess_obs(observation_batch)
        input_batch = tf.concat([pre_obs_batch, action_batch], axis=1)
        return self.__call__(input_batch)

    def _preprocess_obs(self, observation_batch):
        if self._norm_mean is not None and self._norm_stddev is not None:
            return (observation_batch - self.norm_mean) / (self.norm_stddev + 1e-7)
        return observation_batch


class DDPG(tf.keras.Model):
    """Implementation of Twin Delayed DDPG algorithm

    References
    ----------
    Fujimoto, Scott, Herke Hoof, and David Meger. "Addressing function approximation error in actor-critic methods."
    International Conference on Machine Learning. PMLR, 2018.
    """
    def __init__(self, make_actor, make_critic, make_critic2=None,
                 actor_optimizer=tf.keras.optimizers.Adam(1e-3),
                 critic_optimizer=tf.keras.optimizers.Adam(1e-3), gamma=0.99,
                 polyak=0.995, train_actor_noise=0.1, clip_actor_gradients=True,):
        """
        Parameters
        ----------
        make_actor : Function outputting the policy model.
        make_critic : Function outputting the first Q-function model.
        make_critic2 : Function outputting the second Q-function model for double Q-learning, optional.
        actor_optimizer : Optimizer for policy model, default is Adam.
        critic_optimizer : Optimizer for Q-function model, default is Adam.
        gamma : Discount factor, default is 0.99.
        polyak : Polyak update coefficient for target models, default is 0.995.
        train_actor_noise : Noise to utilize for target policy smoothing, default is 0.1.
        clip_actor_gradients : Clip gradients for the policy parameters, default is True.
        """
        super(DDPG, self).__init__()
        self.actor_opt = actor_optimizer
        self.critic_opt = critic_optimizer
        self._act = make_actor()
        self._cri = make_critic()
        self._targ_act = make_actor()
        self._targ_cri = make_critic()
        self._clip_actor_gradients = clip_actor_gradients
        if make_critic2 is not None:
            self._double_q = True
            self._cri2 = make_critic2()
            self._targ_cri2 = make_critic2()
            self._train_cri2 = self._make_critic_train_op(self._cri2, self.critic_opt, gamma, train_actor_noise)
            self._targ_cri2_update = self._make_target_update_op(self._cri2, self._targ_cri2, polyak)
        else:
            self._double_q = False

        self._train_cri = self._make_critic_train_op(self._cri, self.critic_opt, gamma, train_actor_noise)
        self._targ_cri_update = self._make_target_update_op(self._cri, self._targ_cri, polyak)
        self._train_act = self._make_actor_train_op(self.actor_opt)
        self._targ_act_update = self._make_target_update_op(self._act, self._targ_act, polyak)

    def call(self, inputs):
        """Run all models for initialization from observations."""
        out = {}
        out['act'] = self._act.get_action(inputs, 0.0)
        out['t_act'] = self._targ_act.get_action(inputs, 0.0)
        if self._double_q:
            out['q'] = tf.minimum(self._cri.get_q(inputs, out['act']),
                                  self._cri2.get_q(inputs, out['act']))
            out['t_q'] = tf.minimum(self._targ_cri.get_q(inputs, out['act']),
                                    self._targ_cri2.get_q(inputs, out['act']))
        else:
            out['q'] = self._cri.get_q(inputs, out['act'])
            out['t_q'] = self._targ_cri.get_q(inputs, out['act'])
        return out

    def get_action(self, observation_batch, noise_stddev, max_noise=0.5):
        """Return policy action."""
        return self._act.get_action(observation_batch, noise_stddev, max_noise)

    def train(self, buffer, batch_size=128, n_updates=1, act_delay=2):
        """Train policy and Q-function for a given number of steps."""
        for i in range(n_updates):
            b = buffer.get_random_batch(batch_size)
            (observations, actions, next_observations, rewards, done_mask) = (
                b['obs'], b['act'], b['nobs'], b['rew'], b['don'])
            loss_critic = self._train_cri(observations, actions, next_observations, rewards,
                                          done_mask.astype('float32'))
            if self._double_q:
                loss_critic2 = self._train_cri2(observations, actions, next_observations, rewards,
                                                done_mask.astype('float32'))
            if (i + 1) % act_delay == 0:
                loss_actor = self._train_act(observations)
                self._targ_act_update()
                self._targ_cri_update()
                if self._double_q:
                    self._targ_cri2_update()

    def _make_critic_train_op(self, critic, optimizer, discount,
                              noise_stddev):
        if self._double_q:
            def train(observation_batch, action_batch, next_observation_batch,
                      reward_batch, done_mask):
                with tf.GradientTape() as tape:
                    maximizing_action = self._targ_act.get_action(next_observation_batch, noise_stddev)
                    targ_q = tf.minimum(self._targ_cri.get_q(next_observation_batch, maximizing_action),
                                        self._targ_cri2.get_q(next_observation_batch, maximizing_action))
                    targets = tf.reshape(reward_batch, [-1, 1]) + tf.reshape(
                        1 - done_mask, [-1, 1]) * discount * targ_q
                    loss = tf.reduce_mean(tf.square(
                        critic.get_q(observation_batch, action_batch) - targets))
                    gradients = tape.gradient(loss, critic.trainable_weights)
                optimizer.apply_gradients(zip(gradients, critic.trainable_weights))
                return loss
        else:
            def train(observation_batch, action_batch, next_observation_batch,
                      reward_batch, done_mask):
                with tf.GradientTape() as tape:
                    maximizing_action = self._targ_act.get_action(next_observation_batch,
                                                                  noise_stddev)
                    targets = tf.reshape(reward_batch, [-1, 1]) + tf.reshape(
                        1 - done_mask, [-1, 1]) * discount * self._targ_cri.get_q(
                        next_observation_batch, maximizing_action)
                    loss = tf.reduce_mean(tf.square(
                        critic.get_q(observation_batch, action_batch) - targets))
                    gradients = tape.gradient(loss, critic.trainable_weights)
                optimizer.apply_gradients(zip(gradients, critic.trainable_weights))
                return loss
        return tf.function(train)

    def _make_actor_train_op(self, optimizer):
        def train(observation_batch):
            with tf.GradientTape() as tape:
                actions = self._act.get_action(observation_batch, 0.0)
                advantage = self._targ_cri.get_q(observation_batch, actions)
                loss = tf.reduce_mean(-1 * advantage)
                gradients = tape.gradient(loss, self._act.trainable_weights)
                if self._clip_actor_gradients:
                    gradients, _ = tf.clip_by_global_norm(gradients, 40)
            optimizer.apply_gradients(zip(gradients, self._act.trainable_weights))
            return loss

        return tf.function(train)

    def _make_target_update_op(self, model, target_model, polyak):
        def update_target():
            critic_weights = model.trainable_weights
            target_weights = target_model.trainable_weights
            for c_w, t_w in zip(critic_weights, target_weights):
                t_w.assign((polyak) * t_w + (1 - polyak) * c_w)

        return tf.function(update_target)
