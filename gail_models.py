import tensorflow as tf
import tensorflow_probability as tfp


class VisualDiscriminator(tf.keras.layers.Layer):
    """Discriminator with support for visual observations."""
    def __init__(self, layers, stab_const=0.0, rew='mixed'):
        super(VisualDiscriminator, self).__init__()
        self._dis_layers = layers
        self._sb = stab_const
        self._rew = rew

    def call(self, inputs):
        out = inputs
        for layer in self._dis_layers:
            out = layer(out)
        return out

    def get_prob(self, ims):
        model_out = self.__call__(ims)
        return tf.reshape(tf.sigmoid(model_out), [-1])

    def get_reward(self, ims):
        if self._rew == 'positive':
            return -1 * tf.math.log(1 - self.get_prob(ims) + self._sb)
        elif self._rew == 'negative':
            return tf.math.log(self.get_prob(ims) + self._sb)
        return (tf.math.log(self.get_prob(ims) + self._sb) -
                tf.math.log(1 - self.get_prob(ims) + self._sb))


class InvariantDiscriminator(VisualDiscriminator):
    """Invariant discriminator model."""
    def __init__(self, layers, stab_const=0.0, rew='mixed'):
        super(InvariantDiscriminator, self).__init__(layers, stab_const, rew)


class DeterministicPreprocessor(tf.keras.layers.Layer):
    def __init__(self, preprocessing_layers, ):
        super(DeterministicPreprocessor, self).__init__()
        self._pre_layers = preprocessing_layers

    @tf.function
    def call(self, inputs):
        """Get latent representations from visual observations."""
        out = inputs
        for layer in self._pre_layers:
            out = layer(out)
        return out


class GaussianPreprocessor(tf.keras.layers.Layer):
    """Preprocessor outputting Gaussian latent space samples."""
    def __init__(self, preprocessing_layers, scale_stddev=1):
        super(GaussianPreprocessor, self).__init__()
        self._pre_layers = preprocessing_layers
        self._scale_stddev = scale_stddev

    def layers_out(self, inputs):
        out = inputs
        for layer in self._pre_layers:
            out = layer(out)
        return out

    def dist(self, inputs):
        out = self.layers_out(inputs)
        mean, log_stddev = tf.split(out, 2, axis=-1)
        stddev = tf.exp(tf.nn.tanh(log_stddev))*self._scale_stddev
        return tfp.distributions.Normal(loc=mean, scale=stddev)

    @tf.function
    def call(self, inputs):
        """Sample latent representations from visual observations."""
        input_shape = inputs.get_shape()
        out = tf.reshape(inputs, [input_shape[0] * input_shape[1]] + list(input_shape[2:]))
        dist = self.dist(out)
        samples = dist.sample()
        out_shape = samples.get_shape()
        samples = tf.reshape(samples, [input_shape[0], input_shape[1] * out_shape[1]])
        return samples

    @tf.function
    def get_distribution_info(self, inputs):
        """Get Gaussian latent representations distribution parameters from visual observations."""
        input_shape = inputs.get_shape()
        out = tf.reshape(inputs, [input_shape[0] * input_shape[1]] + list(input_shape[2:]))
        out = self.layers_out(out)
        mean, log_stddev = tf.split(out, 2, axis=-1)
        stddev = tf.exp(tf.nn.tanh(log_stddev))*self._scale_stddev
        out_shape = mean.get_shape()
        reshaped_mean = tf.reshape(mean, [input_shape[0], input_shape[1] * out_shape[1]])
        reshaped_stddev = tf.reshape(stddev, [input_shape[0], input_shape[1] * out_shape[1]])
        return reshaped_mean, reshaped_stddev


