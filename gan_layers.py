import tensorflow as tf

tfl = tf.keras.layers


class SpectralNormalization(tfl.Wrapper):
    """Spectral normalization layer wrapper.

    References
    ----------
    Miyato, Takeru, et al. "Spectral normalization for generative adversarial networks." arXiv preprint
    arXiv:1802.05957 (2018).
    """
    def __init__(self, layer, power_iterations=1, eps=1e-12):
        assert isinstance(layer, tf.keras.layers.Layer)
        self.power_iterations = power_iterations
        self._eps = eps
        super(SpectralNormalization, self).__init__(layer)

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
        self.kernel_shape = tf.shape(self.layer.kernel)
        self.u = self.add_weight(shape=[1, self.kernel_shape[-1]],
                                 initializer=tf.keras.initializers.RandomNormal(),
                                 trainable=False)
        self.built = True

    def call(self, inputs):
        self.power_iteration(self.power_iterations)
        return self.layer(inputs)

    def power_iteration(self, iterations):
        reshaped_kernel = tf.reshape(self.layer.kernel, [-1, self.kernel_shape[-1]])
        u = tf.identity(self.u)
        for _ in range(iterations):
            v = tf.matmul(u, tf.transpose(reshaped_kernel))
            v = tf.nn.l2_normalize(v, epsilon=self._eps)
            u = tf.matmul(v, reshaped_kernel)
            u = tf.nn.l2_normalize(u, epsilon=self._eps)
        u, v = tf.stop_gradient(u), tf.stop_gradient(v)
        self.u.assign(u)
        norm_value = tf.matmul(tf.matmul(v, reshaped_kernel), tf.transpose(u))
        self.layer.kernel.assign(self.layer.kernel / norm_value)
