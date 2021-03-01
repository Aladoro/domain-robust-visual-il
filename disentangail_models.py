import tensorflow as tf
from gail_models import GaussianPreprocessor, DeterministicPreprocessor

LN2 = 0.69314718056


class StatisticsNet(tf.keras.layers.Layer):
    """Statistics network model."""

    def __init__(self, mi_layers):
        super(StatisticsNet, self).__init__()
        self._mi_layers = mi_layers

    def layers_out(self, inputs):
        out = inputs
        for layer in self._mi_layers:
            out = layer(out)
        return out

    @tf.function
    def call(self, inputs):
        out = tf.concat(inputs, axis=-1)
        score = self.layers_out(out)
        return score


class MIEstimator(StatisticsNet):
    """Statistics network model for MI estimation."""

    def __init__(self, mi_layers):
        super(MIEstimator, self).__init__(mi_layers)


class DisentanGAIL(tf.keras.Model):
    """Disentangling Generative Adversarial Imitation Learning algorithm."""

    def __init__(self,
                 agent,
                 make_discriminator,
                 make_preprocessing,
                 expert_buffer,
                 prior_expert_buffer=None,
                 prior_agent_buffer=None,
                 make_mi_estimator=None,
                 make_mi2_estimator=None,
                 use_min_double_mi=False,
                 d_loss='ce',
                 d_optimizer=tf.keras.optimizers.Adam(1e-3),
                 mi_optimizer=tf.keras.optimizers.Adam(1e-3),
                 label_smoothing=0.0,
                 stab_const=0.0,
                 mi_constant=0.0,
                 adaptive_mi=False,
                 max_mi=1.0,
                 min_mi=0.5,
                 prior_mi_constant=0.0,
                 max_mi_prior=0.01,
                 negative_priors=False,
                 use_dual_mi=False,
                 mi_lagrangian_optimizer=tf.keras.optimizers.Adam(1e-3),
                 max_mi_constant=10,
                 min_mi_constant=1e-4,
                 min_mi_prior_constant=1e-4,
                 unbiased_mi=False,
                 unbiased_mi_decay=0.99,
                 clip_mi_predictions=False,
                 im_side=32,
                 past_frames=4,):
        """
        Parameters
        ----------
        agent : Imitator agent algorithm.
        make_discriminator : Function outputting the invariant discriminator model.
        make_preprocessing : Function outputting the preprocessor model.
        expert_buffer : Buffer containing the expert demonstrations.
        prior_expert_buffer : Buffer containing the prior expert visual observations, optional.
        prior_agent_buffer : Buffer containing the prior agent visual observations, optional.
        make_mi_estimator : Function outputting the first MI estimator model, optional.
        make_mi2_estimator : Function outputting the second MI estimator model, optional.
        use_min_double_mi : Estimate the MI as the minimum estimate from the two MI estimators, default is False.
        d_loss : Discriminator loss function, 'ce' for cross entropy loss, 'hinge' for hinge loss, default is 'ce'.
        d_optimizer : Optimizer for discriminator model, default is Adam.
        mi_optimizer : Optimizer for MI estimators model, default is Adam.
        label_smoothing : Smoothing coefficient for discriminator loss, default is 0.0.
        stab_const : Stability constant for discriminator loss, default is 0.0.
        mi_constant : Starting expert demonstration constraint penalty, should be greater than 0.0 to penalize
                      preprocessor based on current MI estimate, default is 0.0.
        adaptive_mi : Tune expert demonstration constraint penalty to enforce relative constraint, default is False.
        max_mi : Maximum MI for the expert demonstration constraint regulating the relative penalty, default is 1.0.
        min_mi : Minimum threshold to decrease the expert demonstration constraint penalty, default is 0.5.
        prior_mi_constant : Starting prior data constraint penalty, should be greater than 0.0 to penalize
                            preprocessor based on current MI estimate, default is 0.0.
        max_mi_prior : Maximum MI for the prior data constraint to enforce the dual penalty, default is 0.01.
        negative_priors : Utilize prior data as additional negative examples for discriminator loss, default is False.
        use_dual_mi : Option to use dual penalty to enforce also the expert demonstration contraint, default is False.
        mi_lagrangian_optimizer : Optimizer for enforcing the dual penalty, default is Adam.
        max_mi_constant : Maximum value for the expert demonstration constraint penalty, default is 10.
        min_mi_constant : Minimum value for the expert demonstration constraint penalty, default is 1e-4.
        min_mi_prior_constant : Minimum value for the prior data constraint penalty, default is 1e-4.
        unbiased_mi : Apply the MI estimator bias correction from Belghazi et al. 2018, default is False.
        unbiased_mi_decay : EMA decay for the bias correction from Belghazi et al. 2018, default is 0.99.
        clip_mi_predictions : Clip each mutual information estimate within range [0,1], default is False.
        im_side : Side dimension of the input observations, default is 32.
        past_frames : Number of past visual observations discriminator uses consider for classification, default is 4.
        """

        super(DisentanGAIL, self).__init__()
        self._disc = make_discriminator()
        self._pre = make_preprocessing()
        self._d_loss = d_loss
        self._mi_est = None
        self._unbiased_mi = unbiased_mi
        self._unbiased_mi_decay = unbiased_mi_decay
        self._clip_mi_predictions = clip_mi_predictions
        self._use_min_double_mi = use_min_double_mi
        self._use_dual_mi = use_dual_mi

        if isinstance(self._pre, DeterministicPreprocessor):
            self._lat = False
        elif isinstance(self._pre, GaussianPreprocessor):
            self._lat = True
        else:
            raise NotImplementedError
        self._agent = agent
        self._past_frames = past_frames
        self._exp_buff = expert_buffer
        self._sb = stab_const

        self._adaptive_mi = adaptive_mi
        if self._use_dual_mi:
            self._log_mi_constant = tf.Variable(tf.math.log(mi_constant))
            self._mi_constant = tf.exp(self._log_mi_constant)
            self.update_dual_mi_constant = self._make_dual_mi_constant_update(
                log_mi_constant=self._log_mi_constant,
                max_mi=self._max_mi,
                optimizer=mi_lagrangian_optimizer
            )
            self._log_max_mi_constant = tf.math.log(max_mi_constant)
            self._log_min_mi_constant = tf.math.log(min_mi_constant)
        else:
            self._mi_constant = tf.Variable(mi_constant, trainable=False)
        self._max_mi = max_mi
        self._min_mi = min_mi
        self._max_mi_constant = max_mi_constant
        self._min_mi_constant = min_mi_constant
        if self._adaptive_mi:
            assert self._mi_constant > 0.0, 'When using adaptive mi penalty, ' \
                                            'initialize the mi constant to some ' \
                                            'positive value'
            assert self._max_mi > self._min_mi, 'The maximum MI for the adaptive penalty' \
                                                'should be greater than the minimum MI'
            assert self._max_mi_constant > self._min_mi_constant, 'The maximum MI constant' \
                                                                  'should be greater than ' \
                                                                  'the minimum MI constant'

        if self._mi_constant > 0.0:
            assert make_mi_estimator is not None
            self._mi_est = make_mi_estimator()
            if self._unbiased_mi:
                self._unbiased_mi_ma = tf.Variable(1.0, trainable=False)
            else:
                self._unbiased_mi_ma = None
            self._train_mi = self._make_mi_training_op(self._mi_est, mi_optimizer,
                                                       self._unbiased_mi_ma)
            if make_mi2_estimator is not None:
                self._mi2_est = make_mi2_estimator()
                if self._unbiased_mi:
                    self._unbiased_mi_ma2 = tf.Variable(1.0, trainable=False)
                else:
                    self._unbiased_mi_ma2 = None
                self._train_mi2 = self._make_mi_training_op(self._mi2_est, mi_optimizer,
                                                            self._unbiased_mi_ma2)
                self._double_mi = True
            else:
                self._double_mi = False
        self._mi_prior_constant = prior_mi_constant
        self._max_mi_prior = max_mi_prior

        self._pr_exp_buff = prior_expert_buffer
        self._pr_age_buff = prior_agent_buffer
        if self._mi_prior_constant > 0.0:
            assert self._pr_exp_buff is not None
            assert self._pr_age_buff is not None
            self._log_min_mi_prior_constant = tf.math.log(min_mi_prior_constant)
            self._prior_domains_data = True
            self._log_mi_prior_constant = tf.Variable(tf.math.log(prior_mi_constant))
            self.update_dual_mi_prior_constant = self._make_dual_mi_constant_update(
                log_mi_constant=self._log_mi_prior_constant,
                max_mi=self._max_mi_prior,
                optimizer=mi_lagrangian_optimizer
            )
        else:
            self._prior_domains_data = False
        self._negative_priors = negative_priors
        if self._negative_priors:
            assert self._pr_exp_buff is not None
            assert self._pr_age_buff is not None
        self._train_disc = self._make_disc_training_op(d_optimizer, label_smoothing)

    def _get_expert_pre_batch(self, e_ims_batch):
        return self._pre(e_ims_batch)

    def _get_pre_batch(self, l_ims_batch, e_ims_batch):
        l_pre_batch = self._pre(l_ims_batch)
        e_pre_batch = self._get_expert_pre_batch(e_ims_batch)
        return l_pre_batch, e_pre_batch

    def _get_pre_weights(self, ):
        pre_weights = self._pre.trainable_weights
        return pre_weights

    def _reshape_pre_encodings(self, encodings):
        encodings_shape = tf.shape(encodings)
        return tf.reshape(encodings, [encodings_shape[0] * self._past_frames,
                                      encodings_shape[1] // self._past_frames])

    def call(self, inputs):
        """Run all models for initialization from replay buffer batch."""
        out = self._agent(inputs['obs'])
        out['pre'], out['exp_pre'] = self._get_pre_batch(inputs['ims'], inputs['ims'])
        reshaped_pre = self._reshape_pre_encodings(out['pre'])
        if self._mi_constant > 0.0 or self._adaptive_kl:
            n_inputs = tf.shape(reshaped_pre)[0]
            mi_inputs = tf.concat([reshaped_pre, tf.ones([n_inputs, 1])], axis=1)
            out['mi'] = self._mi_est(mi_inputs)
            if self._double_mi:
                out['mi2'] = self._mi2_est(mi_inputs)
        out['rew'] = self._disc.get_reward(out['pre'])
        return out

    def _gan_loss(self, l_disc_prob, e_disc_prob, lb):
        if self._d_loss == 'hinge':
            return self._hinge_gan_loss(l_disc_prob, e_disc_prob)
        elif self._d_loss == 'ce':
            return self._ce_gan_loss(l_disc_prob, e_disc_prob, lb)
        else:
            raise NotImplementedError

    @staticmethod
    def _hinge_gan_loss(l_disc_prob, e_disc_prob):
        l_disc_loss = tf.nn.relu(1 - l_disc_prob)
        e_disc_loss = tf.nn.relu(1 + e_disc_prob)
        return tf.reduce_mean(l_disc_loss) + tf.reduce_mean(e_disc_loss)

    @staticmethod
    def _ce_gan_loss(l_disc_prob, e_disc_prob, lb):
        labels = tf.concat([tf.zeros_like(l_disc_prob),
                            tf.ones_like(e_disc_prob)], axis=0)
        probs = tf.concat([l_disc_prob, e_disc_prob], axis=0)
        return tf.losses.binary_crossentropy(labels, probs, label_smoothing=lb)

    @staticmethod
    def _dv_kl(est, p_samples, q_samples):
        p_samples_estimate = tf.reduce_mean(est(p_samples))
        q_samples_estimate = tf.math.log(tf.reduce_mean(tf.exp(est(q_samples))))
        return (p_samples_estimate - q_samples_estimate) / LN2

    @staticmethod
    def _get_mi_batches(l_pre_batch, e_pre_batch, past_frames=4):
        l_pre_batch_shape = l_pre_batch.get_shape()
        e_pre_batch_shape = e_pre_batch.get_shape()
        l_pre_batch_n = l_pre_batch_shape[0] * past_frames
        e_pre_batch_n = e_pre_batch_shape[0] * past_frames
        l_pre_batch = tf.reshape(l_pre_batch, [e_pre_batch_n, -1])
        e_pre_batch = tf.reshape(e_pre_batch, [l_pre_batch_n, -1])
        input_correct_batch = tf.concat([l_pre_batch, e_pre_batch], axis=0)
        domain_labels = tf.concat([tf.zeros([l_pre_batch_n, 1]),
                                   tf.ones([e_pre_batch_n, 1])], axis=0)
        shuffled_domain_labels = tf.random.shuffle(domain_labels)
        positive_ordering = tf.concat([input_correct_batch, domain_labels],
                                      axis=1)
        negative_ordering = tf.concat([input_correct_batch, shuffled_domain_labels],
                                      axis=1)
        return positive_ordering, negative_ordering

    def _mi_loss(self, mi_est, l_pre_batch, e_pre_batch):
        positive_ordering, negative_ordering = self._get_mi_batches(
            l_pre_batch, e_pre_batch, self._past_frames)
        return -1 * self._dv_kl(mi_est, positive_ordering, negative_ordering)

    def _make_mi_training_op(self, mi_est, optimizer, mi_ma=None):
        if mi_ma is None:
            def train(l_ims_batch, e_ims_batch):
                with tf.GradientTape() as tape:
                    l_pre_batch, e_pre_batch = self._get_pre_batch(l_ims_batch, e_ims_batch)
                    mi_loss = self._mi_loss(mi_est, l_pre_batch, e_pre_batch)
                    gradients = tape.gradient(mi_loss, mi_est.trainable_weights)
                optimizer.apply_gradients(zip(gradients, mi_est.trainable_weights))
                return mi_loss
        else:
            def loss_fn(mi_est, l_pre_batch, e_pre_batch):
                p_samples, q_samples = self._get_mi_batches(l_pre_batch, e_pre_batch, self._past_frames)
                p_samples_estimate = tf.reduce_mean(mi_est(p_samples))
                batch_q_exp_samples_estimate = tf.reduce_mean(tf.exp(mi_est(q_samples)))
                mi_ma.assign(tf.stop_gradient(self._unbiased_mi_decay * mi_ma +
                                              (1 - self._unbiased_mi_decay) *
                                              batch_q_exp_samples_estimate))
                unbiased_loss = -(p_samples_estimate - batch_q_exp_samples_estimate / mi_ma) / LN2
                mi_loss = -(p_samples_estimate - tf.math.log(batch_q_exp_samples_estimate)) / LN2
                return unbiased_loss, mi_loss

            def train(l_ims_batch, e_ims_batch):
                with tf.GradientTape() as tape:
                    l_pre_batch, e_pre_batch = self._get_pre_batch(l_ims_batch, e_ims_batch)
                    unbiased_loss, mi_loss = loss_fn(mi_est, l_pre_batch, e_pre_batch)
                    gradients = tape.gradient(unbiased_loss, mi_est.trainable_weights)
                optimizer.apply_gradients(zip(gradients, mi_est.trainable_weights))
                return mi_loss
        return tf.function(train)

    def _make_disc_training_op(self, optimizer, lb):
        def compute_losses(l_ims_batch, e_ims_batch, l_prior_ims_batch, e_prior_ims_batch):
            l_pre_batch, e_pre_batch, l_prior_pre_batch, e_prior_pre_batch = gather_pre_batches(l_ims_batch,
                                                                                                e_ims_batch,
                                                                                                l_prior_ims_batch,
                                                                                                e_prior_ims_batch)
            l_disc_prob, e_disc_prob = get_gan_probs(l_pre_batch, e_pre_batch, l_prior_pre_batch,
                                                     e_prior_pre_batch)
            gan_loss = self._gan_loss(l_disc_prob, e_disc_prob, lb)
            mi = get_mi(l_pre_batch, e_pre_batch)
            prior_mi = get_prior_mi(l_prior_pre_batch, e_prior_pre_batch)
            return gan_loss, mi, prior_mi

        # MI calculation
        if self._mi_constant > 0.0:
            if self._use_dual_mi:
                def get_mi_constant():
                    return tf.exp(self._log_mi_constant)
            else:
                def get_mi_constant():
                    return self._mi_constant

            if self._double_mi and self._use_min_double_mi:
                def get_mi(l_pre_batch, e_pre_batch):
                    return tf.math.maximum(
                        -1 * tf.math.minimum(self._mi_loss(self._mi_est, l_pre_batch, e_pre_batch),
                                             self._mi_loss(self._mi2_est, l_pre_batch, e_pre_batch)),
                        0.0)
            else:
                def get_mi(l_pre_batch, e_pre_batch):
                    return tf.math.maximum(-1 * self._mi_loss(self._mi_est, l_pre_batch, e_pre_batch), 0.0)
        else:
            def get_mi_constant():
                return 0.0

            def get_mi(l_pre_batch, e_pre_batch):
                return 0.0

        if self._prior_domains_data or self._negative_priors:
            def gather_pre_batches(l_ims_batch, e_ims_batch, l_prior_ims_batch, e_prior_ims_batch):
                l_pre_batch, e_pre_batch = self._get_pre_batch(l_ims_batch, e_ims_batch)
                l_prior_pre_batch, e_prior_pre_batch = self._get_pre_batch(l_prior_ims_batch, e_prior_ims_batch)
                return l_pre_batch, e_pre_batch, l_prior_pre_batch, e_prior_pre_batch
        else:
            def gather_pre_batches(l_ims_batch, e_ims_batch, l_prior_ims_batch, e_prior_ims_batch):
                l_pre_batch, e_pre_batch = self._get_pre_batch(l_ims_batch, e_ims_batch)
                return l_pre_batch, e_pre_batch, None, None

        # Prior MI calculation
        if self._prior_domains_data:
            def get_prior_mi_constant():
                return tf.exp(self._log_mi_prior_constant)

            if self._double_mi and self._use_min_double_mi:
                def get_prior_mi(l_prior_pre_batch, e_prior_pre_batch):
                    return tf.math.maximum(
                        -1 * tf.math.minimum(self._mi_loss(self._mi_est, l_prior_pre_batch, e_prior_pre_batch),
                                             self._mi_loss(self._mi2_est, l_prior_pre_batch, e_prior_pre_batch)),
                        0.0)
            else:
                def get_prior_mi(l_prior_pre_batch, e_prior_pre_batch):
                    return tf.math.maximum(-1 * self._mi_loss(self._mi_est, l_prior_pre_batch, e_prior_pre_batch), 0.0)
        else:
            def get_prior_mi_constant():
                return 0.0

            def get_prior_mi(l_prior_ims_batch, e_prior_ims_batch):
                return 0.0

        if self._negative_priors:
            def get_gan_probs(l_pre_batch, e_pre_batch, l_prior_pre_batch, e_prior_pre_batch):
                l_disc_prob = self._disc.get_prob(l_pre_batch) + self._sb
                e_disc_prob = self._disc.get_prob(e_pre_batch) + self._sb
                l_prior_disc_prob = self._disc.get_prob(l_prior_pre_batch) + self._sb
                e_prior_disc_prob = self._disc.get_prob(e_prior_pre_batch) + self._sb
                return tf.concat([l_disc_prob, l_prior_disc_prob, e_prior_disc_prob], axis=0), e_disc_prob
        else:
            def get_gan_probs(l_pre_batch, e_pre_batch, l_prior_pre_batch, e_prior_pre_batch):
                l_disc_prob = self._disc.get_prob(l_pre_batch) + self._sb
                e_disc_prob = self._disc.get_prob(e_pre_batch) + self._sb
                return l_disc_prob, e_disc_prob

        def get_gradients(l_ims_batch, e_ims_batch, l_prior_ims_batch, e_prior_ims_batch):
            with tf.GradientTape(persistent=True) as tape:
                gan_loss, mi, prior_mi = compute_losses(l_ims_batch=l_ims_batch,
                                                        e_ims_batch=e_ims_batch,
                                                        l_prior_ims_batch=l_prior_ims_batch,
                                                        e_prior_ims_batch=e_prior_ims_batch)
                weighted_mi = mi * get_mi_constant() + get_prior_mi_constant() * prior_mi
                loss = gan_loss + weighted_mi
            gradients = tape.gradient(loss, self._disc.trainable_weights + self._get_pre_weights())
            del tape
            return gradients, gan_loss

        def train(l_ims_batch, e_ims_batch, l_prior_ims_batch=None, e_prior_ims_batch=None):
            gradients, gan_loss = get_gradients(l_ims_batch=l_ims_batch,
                                                e_ims_batch=e_ims_batch,
                                                l_prior_ims_batch=l_prior_ims_batch,
                                                e_prior_ims_batch=e_prior_ims_batch, )
            optimizer.apply_gradients(zip(gradients, self._disc.trainable_weights + self._get_pre_weights()))
            return gan_loss

        return tf.function(train)

    @staticmethod
    def _make_dual_mi_constant_update(log_mi_constant, max_mi, optimizer):
        def update_dual_mi_constant(mi_estimate):
            mi_diff = max_mi - mi_estimate
            with tf.GradientTape() as tape:
                mi_dual_loss = log_mi_constant * tf.stop_gradient(mi_diff)
                gradients = tape.gradient(mi_dual_loss, [log_mi_constant])
            optimizer.apply_gradients(zip(gradients, [log_mi_constant]))

        return update_dual_mi_constant

    def _get_random_im_batches(self,
                               agent_buffer,
                               d_e_batch_size,
                               d_l_batch_size, ):
        l_batch = agent_buffer.get_random_batch(d_l_batch_size, False)
        l_ims = l_batch['ims']
        e_batch = self._exp_buff.get_random_batch(d_e_batch_size)
        e_ims = e_batch['ims']
        return l_ims, e_ims

    def _get_random_im_prior_batches(self,
                                     d_e_batch_size,
                                     d_l_batch_size, ):
        if self._prior_domains_data or self._negative_priors:
            l_batch = self._pr_age_buff.get_random_batch(d_l_batch_size)
            l_ims = l_batch['ims']
            e_batch = self._pr_exp_buff.get_random_batch(d_e_batch_size)
            e_ims = e_batch['ims']
            return l_ims, e_ims
        else:
            return None, None

    def train(self, agent_buffer, l_batch_size=128, l_updates=1, l_act_delay=1,
              d_updates=1, mi_updates=1, d_e_batch_size=128, d_l_batch_size=128):
        """Train discriminator, statistics network, and learner agent models.

        Parameters
        ----------
        agent_buffer : Buffer containing the agent-collected experience.
        l_batch_size : Batch size of agent experience used to train the learner agent models, default is 128.
        l_updates : Number of updates to train learner agent, default is 1.
        l_act_delay : Actor delay (1/frequency) to train the learner agent policy, default is 1.
        d_updates : Number of updates to train the discriminator, default is 1.
        mi_updates : Number of updates to train the statistics network, default is 1.
        d_e_batch_size : Batch size of agent experience used to train the discriminator models, default is 128.
        d_l_batch_size : Batch size of expert experience used to train the discriminator models, default is 128.
        """
        mi = []
        pr_mi = []
        if self._mi_constant > 0:
            m_updates_per_d = mi_updates // d_updates
            assert m_updates_per_d > 0, 'The number of MINE updates should be at least the number of discriminator ' \
                                        'updates'
        for _ in range(d_updates):
            l_ims, e_ims = self._get_random_im_batches(agent_buffer=agent_buffer,
                                                       d_e_batch_size=d_e_batch_size,
                                                       d_l_batch_size=d_l_batch_size, )
            l_prior_ims, e_prior_ims = self._get_random_im_prior_batches(d_e_batch_size=d_e_batch_size,
                                                                         d_l_batch_size=d_l_batch_size)
            c_gan_loss = self._train_disc(l_ims, e_ims, l_prior_ims, e_prior_ims)
            if self._mi_constant > 0.0 or self._adaptive_kl or self._prior_domains_data:
                for _ in range(m_updates_per_d):
                    l_ims, e_ims = self._get_random_im_batches(agent_buffer=agent_buffer,
                                                               d_e_batch_size=d_e_batch_size,
                                                               d_l_batch_size=d_l_batch_size)
                    mi_loss = self._train_mi(l_ims, e_ims)
                    if self._prior_domains_data:
                        l_ims, e_ims = self._get_random_im_prior_batches(d_e_batch_size=d_e_batch_size,
                                                                         d_l_batch_size=d_l_batch_size)
                        mi_prior_loss = self._train_mi(l_ims, e_ims)
                        mi_prior_est = -1 * mi_prior_loss
                    if self._double_mi:
                        l_ims, e_ims = self._get_random_im_batches(agent_buffer=agent_buffer,
                                                                   d_e_batch_size=d_e_batch_size,
                                                                   d_l_batch_size=d_l_batch_size)
                        mi2_loss = self._train_mi2(l_ims, e_ims)
                        mi_est = -1 * tf.minimum(mi_loss, mi2_loss)
                        if self._clip_mi_predictions:
                            mi_est = tf.clip_by_value(mi_est, 0.0, 1.0)
                        mi.append(mi_est)
                        if self._prior_domains_data:
                            l_ims, e_ims = self._get_random_im_prior_batches(d_e_batch_size=d_e_batch_size,
                                                                             d_l_batch_size=d_l_batch_size)
                            mi2_prior_loss = self._train_mi2(l_ims, e_ims)
                            mi2_prior_est = -1 * mi2_prior_loss
                            mi_prior_est = tf.maximum(mi_prior_est, mi2_prior_est)
                            if self._clip_mi_predictions:
                                mi_prior_est = tf.clip_by_value(mi_prior_est, 0.0, 1.0)
                            pr_mi.append(mi_prior_est)
                    else:
                        mi_est = -1 * mi_loss
                        if self._clip_mi_predictions:
                            mi_est = tf.clip_by_value(mi_est, 0.0, 1.0)
                        mi.append(mi_est)
                        if self._prior_domains_data:
                            if self._clip_mi_predictions:
                                mi_prior_est = tf.clip_by_value(mi_prior_est, 0.0, 1.0)
                            pr_mi.append(mi_prior_est)
                if self._use_dual_mi:
                    self.update_dual_mi_constant(mi_est)
                    self._log_mi_constant.assign(tf.clip_by_value(self._log_mi_constant,
                                                                  self._log_min_mi_constant,
                                                                  self._log_max_mi_constant))
                    self._mi_constant = tf.exp(self._log_mi_constant)
                if self._prior_domains_data:
                    self.update_dual_mi_prior_constant(mi_prior_est)
                    self._log_mi_prior_constant.assign(tf.maximum(self._log_mi_prior_constant,
                                                                  self._log_min_mi_prior_constant))
                    self._mi_prior_constant = tf.exp(self._log_mi_prior_constant)

        if self._mi_constant > 0.0:
            average_mi = tf.reduce_mean(mi)
        if self._adaptive_mi:
            if not self._use_dual_mi:
                if average_mi > self._max_mi:
                    self._mi_constant.assign(self._mi_constant * 1.5)
                elif average_mi < self._min_mi:
                    self._mi_constant.assign(self._mi_constant / 1.5)
                self._mi_constant.assign(tf.clip_by_value(self._mi_constant,
                                                          self._min_mi_constant,
                                                          self._max_mi_constant))
        self._agent.train(agent_buffer, l_batch_size, l_updates, l_act_delay)
