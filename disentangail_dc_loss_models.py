import tensorflow as tf


class DomainConfusionDisentanGAIL(tf.keras.Model):
    """DisentanGAIL algorithm with domain confusion loss instead of proposed constraints.

    References
    ----------
    Stadie, Bradly C., Pieter Abbeel, and Ilya Sutskever. "Third-person imitation learning." arXiv preprint
    arXiv:1703.01703 (2017).
    """
    def __init__(self,
                 agent,
                 make_discriminator,
                 make_preprocessing,
                 expert_buffer,
                 prior_expert_buffer=None,
                 prior_agent_buffer=None,
                 d_optimizer=tf.keras.optimizers.Adam(1e-3),
                 d_domain_constant=0.25,
                 stab_const=0.0,
                 past_frames=4,):
        """

        Parameters
        ----------
        agent : Imitator agent algorithm.
        make_discriminator : Function outputting the domain and class discriminator models.
        make_preprocessing : Function outputting the feature extractor model.
        expert_buffer : Buffer containing the expert demonstrations.
        prior_expert_buffer : Buffer containing the prior expert visual observations, optional.
        prior_agent_buffer : Buffer containing the prior agent visual observations, optional.
        d_optimizer : Optimizer for discriminator model, default is Adam.
        d_domain_constant : Domain confusion loss constant, default is 0.25.
        stab_const : Stability constant for discriminator loss, default is 0.0.
        past_frames : Number of past visual observations discriminator uses consider for classification, default is 4.
        """

        super(DomainConfusionDisentanGAIL, self).__init__()
        self._disc = make_discriminator()
        self._domain_disc = make_discriminator()
        self._pre = make_preprocessing()
        self._domain_constant = tf.constant(d_domain_constant)

        self._agent = agent
        self._past_frames = past_frames
        self._exp_buff = expert_buffer
        self._sb = stab_const

        self._prior_exp_buff = prior_expert_buffer
        self._prior_age_buff = prior_agent_buffer

        if self._prior_exp_buff is not None and self._prior_age_buff is not None:
            self._prior_data = True
        else:
            self._prior_data = False

        self._train_disc = self._make_disc_training_op(d_optimizer)

    def _get_pre_batch(self, l_ims_batch, e_ims_batch):
        l_pre_batch = self._pre(l_ims_batch)
        e_pre_batch = self._pre(e_ims_batch)
        return l_pre_batch, e_pre_batch

    def _get_pre_weights(self, ):
        pre_weights = self._pre.trainable_weights
        return pre_weights

    def _reshape_pre_encodings(self, encodings):
        encodings_shape = tf.shape(encodings)
        return tf.reshape(encodings, [encodings_shape[0] * self._past_frames,
                                      encodings_shape[1] // self._past_frames])

    def call(self, inputs):
        out = self._agent(inputs['obs'])
        out['pre'], out['exp_pre'] = self._get_pre_batch(inputs['ims'], inputs['ims'])
        reshaped_pre = self._reshape_pre_encodings(out['pre'])
        out['rew'] = self._disc.get_reward(out['pre'])
        out['class_score'] = self._domain_disc.get_prob(reshaped_pre)
        return out

    def _gan_loss(self, l_disc_prob, e_disc_prob):
        labels = tf.concat([tf.zeros_like(l_disc_prob),
                            tf.ones_like(e_disc_prob)], axis=0)
        probs = tf.concat([l_disc_prob, e_disc_prob], axis=0)
        return tf.losses.binary_crossentropy(labels, probs)

    def _make_disc_training_op(self, optimizer):
        if self._prior_data:
            def compute_losses(l_ims_batch, e_ims_batch, l_prior_ims_batch, e_prior_ims_batch):
                l_pre_batch, e_pre_batch = self._get_pre_batch(l_ims_batch, e_ims_batch)
                l_prior_pre_batch, e_prior_pre_batch = self._get_pre_batch(l_prior_ims_batch, e_prior_ims_batch)
                l_disc_prob = self._disc.get_prob(l_pre_batch) + self._sb
                e_disc_prob = self._disc.get_prob(e_pre_batch) + self._sb
                l_prior_prob = self._disc.get_prob(l_prior_pre_batch) + self._sb
                e_prior_prob = self._disc.get_prob(e_prior_pre_batch) + self._sb
                disc_loss = tf.reduce_mean(self._gan_loss(tf.concat([l_disc_prob, l_prior_prob, e_prior_prob], axis=0),
                                                          e_disc_prob))
                l_pre_batch_reshaped = self._reshape_pre_encodings(l_pre_batch)
                e_pre_batch_reshaped = self._reshape_pre_encodings(e_pre_batch)
                l_prior_pre_batch_reshaped = self._reshape_pre_encodings(l_prior_pre_batch)
                e_prior_pre_batch_reshaped = self._reshape_pre_encodings(e_prior_pre_batch)

                l_domain_disc_prob = self._domain_disc.get_prob(l_pre_batch_reshaped) + self._sb
                e_domain_disc_prob = self._domain_disc.get_prob(e_pre_batch_reshaped) + self._sb
                l_prior_domain_disc_prob = self._domain_disc.get_prob(l_prior_pre_batch_reshaped) + self._sb
                e_prior_domain_disc_prob = self._domain_disc.get_prob(e_prior_pre_batch_reshaped) + self._sb

                domain_loss = tf.reduce_mean((self._domain_constant *
                                              self._gan_loss(tf.concat([l_domain_disc_prob, l_prior_domain_disc_prob],
                                                                       axis=0),
                                                             tf.concat([e_domain_disc_prob, e_prior_domain_disc_prob],
                                                                      axis=0))))
                return disc_loss, domain_loss
        else:
            def compute_losses(l_ims_batch, e_ims_batch, l_prior_ims_batch, e_prior_ims_batch):
                l_pre_batch, e_pre_batch = self._get_pre_batch(l_ims_batch, e_ims_batch)
                l_disc_prob = self._disc.get_prob(l_pre_batch) + self._sb
                e_disc_prob = self._disc.get_prob(e_pre_batch) + self._sb
                disc_loss = tf.reduce_mean(self._gan_loss(l_disc_prob, e_disc_prob))

                l_pre_batch_reshaped = self._reshape_pre_encodings(l_pre_batch)
                e_pre_batch_reshaped = self._reshape_pre_encodings(e_pre_batch)

                l_domain_disc_prob = self._domain_disc.get_prob(l_pre_batch_reshaped) + self._sb
                e_domain_disc_prob = self._domain_disc.get_prob(e_pre_batch_reshaped) + self._sb

                domain_loss = tf.reduce_mean((self._domain_constant * self._gan_loss(l_domain_disc_prob,
                                                                                     e_domain_disc_prob)))
                return disc_loss, domain_loss

        def get_gradients(l_ims_batch, e_ims_batch, l_prior_ims_batch, e_prior_ims_batch):
            with tf.GradientTape(persistent=True) as tape:
                disc_loss, domain_loss = compute_losses(l_ims_batch, e_ims_batch, l_prior_ims_batch, e_prior_ims_batch)
            disc_loss_gradients = tape.gradient(disc_loss, self._disc.trainable_weights +
                                                self._pre.trainable_weights)
            disc_loss_disc_gradients = disc_loss_gradients[:len(self._disc.trainable_weights)]
            disc_loss_pre_gradients = disc_loss_gradients[len(self._disc.trainable_weights):]
            domain_loss_gradients = tape.gradient(domain_loss, self._domain_disc.trainable_weights +
                                                  self._pre.trainable_weights)
            domain_loss_disc_gradients = domain_loss_gradients[:len(self._domain_disc.trainable_weights)]
            domain_loss_pre_gradients = domain_loss_gradients[len(self._domain_disc.trainable_weights):]
            pre_gradients = [disc_loss_pre_g + domain_loss_pre_g for disc_loss_pre_g, domain_loss_pre_g in
                             zip(disc_loss_pre_gradients, domain_loss_pre_gradients)]
            gradients = disc_loss_disc_gradients + domain_loss_disc_gradients + pre_gradients
            losses = (disc_loss, domain_loss)
            del tape
            return gradients, losses

        def train(l_ims_batch, e_ims_batch, l_prior_ims_batch=None, e_prior_ims_batch=None):
            gradients, losses = get_gradients(l_ims_batch, e_ims_batch, l_prior_ims_batch, e_prior_ims_batch)
            optimizer.apply_gradients(zip(gradients, self._disc.trainable_weights + self._domain_disc.trainable_weights +
                                          self._pre.trainable_weights))
            disc_loss, domain_loss = losses
            return disc_loss, domain_loss

        return tf.function(train)

    def _get_random_im_batches(self,
                               dac_buffer,
                               d_e_batch_size,
                               d_l_batch_size,):
        l_batch = dac_buffer.get_random_batch(d_l_batch_size, False)
        l_ims = l_batch['ims']
        e_batch = self._exp_buff.get_random_batch(d_e_batch_size)
        e_ims = e_batch['ims']
        return l_ims, e_ims

    def _get_random_im_prior_batches(self,
                                     d_e_batch_size,
                                     d_l_batch_size,):
        if self._prior_data:
            l_batch = self._prior_age_buff.get_random_batch(d_l_batch_size)
            l_ims = l_batch['ims']
            e_batch = self._prior_exp_buff.get_random_batch(d_e_batch_size)
            e_ims = e_batch['ims']
            return l_ims, e_ims
        else:
            return None, None

    def train(self, agent_buffer, l_batch_size=128, l_updates=1, l_act_delay=1,
              d_updates=1, d_e_batch_size=128, d_l_batch_size=128,):
        """Train class discriminator, domain discriminator, and learner agent models.

        Parameters
        ----------
        agent_buffer : Buffer containing the agent-collected experience.
        l_batch_size : Batch size of agent experience used to train the learner agent models, default is 128.
        l_updates : Number of updates to train learner agent, default is 1.
        l_act_delay : Actor delay (1/frequency) to train the learner agent policy, default is 1.
        d_updates : Number of updates to train the discriminators, default is 1.
        d_e_batch_size : Batch size of agent experience used to train the discriminator models, default is 128.
        d_l_batch_size : Batch size of expert experience used to train the discriminator models, default is 128.
        """
        for _ in range(d_updates):
            l_ims, e_ims = self._get_random_im_batches(dac_buffer=agent_buffer,
                                                       d_e_batch_size=d_e_batch_size,
                                                       d_l_batch_size=d_l_batch_size, )
            l_prior_ims, e_prior_ims = self._get_random_im_prior_batches(d_e_batch_size=d_e_batch_size,
                                                                         d_l_batch_size=d_l_batch_size)
            disc_loss, domain_loss = self._train_disc(l_ims, e_ims, l_prior_ims, e_prior_ims)
        self._agent.train(agent_buffer, l_batch_size, l_updates, l_act_delay)
