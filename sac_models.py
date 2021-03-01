import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

LN4 = np.log(4)


class StochasticActor(tf.keras.layers.Layer):
    """Stochastic policy model."""
    def __init__(self, layers, norm_mean=None, norm_stddev=None):
        super(StochasticActor, self).__init__()
        self._act_layers = layers
        self._out_dim = layers[-1].units
        self._norm_mean = norm_mean
        self._norm_stddev = norm_stddev

    def call(self, inputs):
        """Compute the policy output distribution from observations."""
        out = inputs
        for layer in self._act_layers:
            out = layer(out)
        mean, log_stddev = tf.split(out, 2, -1)
        stddev = tf.exp(log_stddev)
        return tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=stddev)

    def get_action(self, observation_batch, noise_stddev, *args, **kwargs):
        """Compute a single action sample from observations."""
        pre_obs_batch = self._preprocess_obs(observation_batch)
        distribution = self.__call__(pre_obs_batch)
        if noise_stddev == 0.0:
            return tf.tanh(distribution.mean())
        return tf.tanh(distribution.sample())

    def get_action_probability(self, observation_batch):
        """Compute a single action sample and the relative probabilities from observations."""
        pre_obs_batch = self._preprocess_obs(observation_batch)
        distribution = self.__call__(pre_obs_batch)
        raw_actions = distribution.sample()
        actions = tf.tanh(raw_actions)
        log_probs = distribution.log_prob(raw_actions)
        squash_features = -2 * raw_actions
        squash_correction = tf.reduce_sum(LN4 + squash_features - 2 * tf.math.softplus(squash_features), axis=1)
        log_probs -= squash_correction
        log_probs = tf.reshape(log_probs, [-1, 1])
        return actions, log_probs

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


class SAC(tf.keras.Model):
    """Implementation of Soft Actor-Critic algorithm

    References
    ----------
    Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and applications." arXiv preprint arXiv:1812.05905 (2018).
    """
    def __init__(self, make_actor, make_critic, make_critic2=None, actor_optimizer=tf.keras.optimizers.Adam(1e-3),
                 critic_optimizer=tf.keras.optimizers.Adam(1e-3), gamma=0.99, polyak=0.995, entropy_coefficient=0.1,
                 tune_entropy_coefficient=False, target_entropy=-6, clip_actor_gradients=True,):
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
        entropy_coefficient : Starting SAC entropy coefficient, default is 0.1.
        tune_entropy_coefficient : Automatically tune entropy coefficient, default is False.
        target_entropy : Target entropy used when automatically tuning entropy coefficient, default is -6.
        clip_actor_gradients : Clip gradients for the policy parameters, default is True.
        """
        super(SAC, self).__init__()
        self.actor_opt = actor_optimizer
        self.critic_opt = critic_optimizer
        self._entropy_coefficient = entropy_coefficient
        self._tune_entropy_coefficient = tune_entropy_coefficient
        self._target_entropy = float(target_entropy)
        self._act = make_actor()
        self._cri = make_critic()
        self._targ_cri = make_critic()
        self._clip_actor_gradients = clip_actor_gradients
        if make_critic2 is not None:
            self._double_q = True
            self._cri2 = make_critic2()
            self._targ_cri2 = make_critic2()
            self._train_cri2 = self._make_critic_train_op(self._cri2, self.critic_opt, gamma)
            self._targ_cri2_update = self._make_target_update_op(self._cri2, self._targ_cri2, polyak)
        else:
            self._double_q = False

        self._train_cri = self._make_critic_train_op(self._cri, self.critic_opt, gamma)
        self._targ_cri_update = self._make_target_update_op(self._cri, self._targ_cri, polyak)
        self._train_act = self._make_actor_train_op(self.actor_opt)

        if self._tune_entropy_coefficient:
            self._log_alpha = tf.Variable(0.0, trainable=True)
        else:
            self._log_alpha = tf.Variable(tf.math.log(
                self._entropy_coefficient), trainable=False)

        self._train_alpha = self._make_alpha_train_op(self.actor_opt)

    def call(self, inputs):
        """Run all models for initialization from observations."""
        out = {}
        out['act'] = self._act.get_action(inputs, 0.0)
        if self._double_q:
            out['q'] = tf.minimum(self._cri.get_q(inputs, out['act']), self._cri2.get_q(inputs, out['act']))
            out['t_q'] = tf.minimum(self._targ_cri.get_q(inputs, out['act']), self._targ_cri2.get_q(inputs, out['act']))
        else:
            out['q'] = self._cri.get_q(inputs, out['act'])
            out['t_q'] = self._targ_cri.get_q(inputs, out['act'])
        return out

    def get_action(self, observation_batch, noise_stddev, max_noise=0.5):
        """Return policy action."""
        return self._act.get_action(observation_batch, noise_stddev, max_noise)

    def train(self, buffer, batch_size=128, n_updates=1, act_delay=1):
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
                loss_alpha = self._train_alpha(observations)
                self._targ_cri_update()
                if self._double_q:
                    self._targ_cri2_update()

    def _make_critic_train_op(self, critic, optimizer, discount):
        if self._double_q:
            def q_estimator(observations, actions):
                q_1 = self._targ_cri.get_q(observations, actions)
                q_2 = self._targ_cri2.get_q(observations, actions)
                return tf.minimum(q_1, q_2)
        else:
            def q_estimator(observations, actions):
                return self._targ_cri.get_q(observations, actions)

        def train(observation_batch, action_batch, next_observation_batch,
                  reward_batch, done_mask):
            with tf.GradientTape() as tape:
                next_actions, next_log_probs = self._act.get_action_probability(
                    next_observation_batch)
                next_q = q_estimator(next_observation_batch, next_actions)
                targets = tf.reshape(reward_batch, [-1, 1]) + tf.reshape(
                    1 - done_mask, [-1, 1]) * discount * (
                                  next_q - tf.exp(self._log_alpha) * next_log_probs)
                loss = 0.5 * tf.reduce_mean(tf.square(
                    critic.get_q(observation_batch, action_batch) - tf.stop_gradient(targets)))
                gradients = tape.gradient(loss, critic.trainable_weights)
            optimizer.apply_gradients(zip(gradients, critic.trainable_weights))
            return loss

        return tf.function(train)

    def _make_actor_train_op(self, optimizer):
        if self._double_q:
            def q_estimator(observations, actions):
                q_1 = self._cri.get_q(observations, actions)
                q_2 = self._cri2.get_q(observations, actions)
                return tf.minimum(q_1, q_2)
        else:
            def q_estimator(observations, actions):
                return self._cri.get_q(observations, actions)

        def train(observation_batch):
            with tf.GradientTape() as tape:
                actions, log_probs = self._act.get_action_probability(observation_batch)
                q_estimates = q_estimator(observation_batch, actions)
                loss = tf.exp(self._log_alpha) * log_probs - q_estimates
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

    def _make_alpha_train_op(self, optimizer):
        if self._tune_entropy_coefficient:
            def train(observation_batch):
                with tf.GradientTape() as tape:
                    actions, log_probs = self._act.get_action_probability(observation_batch)
                    loss = -tf.reduce_mean(self._log_alpha * tf.stop_gradient(
                        (log_probs + self._target_entropy)))
                    gradients = tape.gradient(loss, [self._log_alpha])
                optimizer.apply_gradients(zip(gradients, [self._log_alpha]))
                return loss
        else:
            def train(observation_batch):
                return 0.0
        return tf.function(train)
