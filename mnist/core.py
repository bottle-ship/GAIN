import os

import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import backend as kb
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras.utils import plot_model
from tqdm import trange


def make_directory(path, time_suffix=False):
    if time_suffix:
        path += datetime.datetime.now().strftime("_%Y%m%d-%H%M%S")

    if os.path.exists(path):
        raise FileExistsError("'%s' is already exists." % path)
    else:
        path = path.split(os.sep)
        for i in range(0, len(path)):
            if not os.path.exists(os.sep.join(path[:i + 1])):
                os.mkdir(os.sep.join(path[:i + 1]))
        path = os.sep.join(path)

        return path


def load_model_from_json(filename):
    with open(filename, "r") as json_file:
        loaded_json_file = json_file.read()

    return models.model_from_json(loaded_json_file)


def save_model_to_json(model, filename):
    model_json = model.to_json()
    with open(filename, "w") as model_file:
        model_file.write(model_json)


class GAIN(object):

    def __init__(self, input_shape,
                 batch_size=128,
                 alpha=10,
                 tf_verbose=False):
        if not tf_verbose:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        tf.compat.v1.enable_eager_execution()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        kb.set_session(tf.compat.v1.Session(config=config))

        self.input_shape = input_shape
        self.batch_size = batch_size
        self.alpha = alpha

        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()

        self._g_optimizer = tf.keras.optimizers.Adam()
        self._d_optimizer = tf.keras.optimizers.Adam()

        self.p_miss = 0.5
        self.p_hint = 0.9

    def _build_generator(self):
        z = layers.Input(shape=self.input_shape)
        m = layers.Input(shape=self.input_shape)

        inputs = layers.concatenate([z, m])

        fc1 = layers.Dense(256, kernel_initializer='glorot_uniform', bias_initializer='zeros')(inputs)
        fc1 = layers.ReLU()(fc1)

        fc2 = layers.Dense(128, kernel_initializer='glorot_uniform', bias_initializer='zeros')(fc1)
        fc2 = layers.ReLU()(fc2)

        fc3 = layers.Dense(784, kernel_initializer='glorot_uniform', bias_initializer='zeros')(fc2)

        outputs = layers.Activation('sigmoid')(fc3)

        return models.Model([z, m], outputs)

    def _build_discriminator(self):
        g = layers.Input(shape=self.input_shape)
        h = layers.Input(shape=self.input_shape)

        inputs = layers.concatenate([g, h])

        fc1 = layers.Dense(256, kernel_initializer='glorot_uniform', bias_initializer='zeros')(inputs)
        fc1 = layers.ReLU()(fc1)

        fc2 = layers.Dense(128, kernel_initializer='glorot_uniform', bias_initializer='zeros')(fc1)
        fc2 = layers.ReLU()(fc2)

        fc3 = layers.Dense(784, kernel_initializer='glorot_uniform', bias_initializer='zeros')(fc2)

        outputs = layers.Activation('sigmoid')(fc3)

        return models.Model([g, h], outputs)

    def _compute_gradients(self, z, m, h):
        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            g_out = self.generator([z, m], training=True)
            hat_g = z * m + g_out * (1 - m)
            d = self.discriminator([hat_g, h], training=True)

            d_loss = -tf.reduce_mean(m * tf.math.log(d + 1e-8) + (1 - m) * tf.math.log(1. - d + 1e-8)) * 2

            g_loss_entropy = -tf.reduce_mean((1 - m) * tf.math.log(d + 1e-8)) / tf.reduce_mean(1 - m)
            # g_loss_mse = tf.reduce_mean((m * z - m * g_out)**2) / tf.reduce_mean(m)
            g_loss_mse = losses.MeanSquaredError()(m * z, m * g_out)
            g_loss = g_loss_entropy + self.alpha * g_loss_mse

            d_grad = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
            g_grad = g_tape.gradient(g_loss, self.generator.trainable_variables)

            return d_grad, g_grad, d_loss, g_loss, g_loss_mse

    def _apply_gradients_generator(self, grad_generator):
        self._g_optimizer.apply_gradients(zip(grad_generator, self.generator.trainable_variables))

    def _apply_gradients_discriminator(self, grad_discriminator):
        self._d_optimizer.apply_gradients(zip(grad_discriminator, self.discriminator.trainable_variables))

    @staticmethod
    def _get_random_idx(x, n):
        return np.random.permutation(x.shape[0])[:n]

    @staticmethod
    def _get_random_noise(x):
        return np.random.uniform(0., 1., size=x.shape)

    @staticmethod
    def _get_mask_vector(x, p):
        a = np.random.uniform(0., 1., size=x.shape)
        b = a > p
        c = 1. * b

        return c

    def _plot_validation(self, x, iteration, log_dir):
        m = self._get_mask_vector(x, self.p_miss)

        batch_idx = self._get_random_idx(x, 5)
        x_batch = x[batch_idx, ...]
        m_batch = m[batch_idx, ...]
        z_batch = m_batch * x_batch + (1 - m_batch) * self._get_random_noise(x_batch)

        fig = plt.figure(figsize=(5, 5))
        gs = gridspec.GridSpec(5, 5)
        gs.update(wspace=0.05, hspace=0.05)

        z_val_batch_1 = m_batch * x_batch + (1 - m_batch) * self._get_random_noise(x_batch)
        samples1 = self.generator([z_val_batch_1, m_batch], training=False)

        z_val_batch_2 = m_batch * x_batch + (1 - m_batch) * self._get_random_noise(x_batch)
        samples2 = self.generator([z_val_batch_2, m_batch], training=False)

        z_val_batch_3 = m_batch * x_batch + (1 - m_batch) * self._get_random_noise(x_batch)
        samples3 = self.generator([z_val_batch_3, m_batch], training=False)

        samples = np.vstack([z_batch, samples1, samples2, samples3, x_batch])

        for i, sample in enumerate(samples):
            ax = fig.add_subplot(gs[i])
            ax.axis('off')
            ax.imshow(sample.reshape(28, 28), cmap='Greys_r')

        plt.savefig(os.path.join(log_dir, '%05d.png' % iteration), bbox_inches='tight')
        plt.close(fig)

    def fit(self, x, validation=None, iterations=10000, log_dir=None):
        if log_dir is not None:
            log_dir = make_directory(log_dir, time_suffix=True)
            self.show_discriminator_model(os.path.join(log_dir, 'discriminator.png'))
            self.show_generator_model(os.path.join(log_dir, 'generator.png'))

        m = self._get_mask_vector(x, self.p_miss)

        with trange(iterations) as tqdm_range:
            for iteration in tqdm_range:
                batch_idx = self._get_random_idx(x, self.batch_size)
                x_batch = x[batch_idx, ...]
                m_batch = m[batch_idx, ...]

                z_batch = m_batch * x_batch + (1 - m_batch) * self._get_random_noise(x_batch)
                h_batch = m_batch * self._get_mask_vector(x_batch, 1 - self.p_hint)

                z_batch = tf.convert_to_tensor(z_batch, dtype=tf.dtypes.float32)
                m_batch = tf.convert_to_tensor(m_batch, dtype=tf.dtypes.float32)
                h_batch = tf.convert_to_tensor(h_batch, dtype=tf.dtypes.float32)

                d_grad, g_grad, d_loss, g_loss, g_loss_mse = self._compute_gradients(z=z_batch, m=m_batch, h=h_batch)

                self._apply_gradients_discriminator(d_grad)
                self._apply_gradients_generator(g_grad)

                tqdm_range.set_postfix_str("[Loss D] %.3f [Loss G] %.3f [Loss MSE] %.3f" % (d_loss, g_loss, g_loss_mse))

                if iteration % 100 == 0 and validation is not None and log_dir is not None:
                    self._plot_validation(validation, iteration, log_dir)

        return self

    def show_generator_model(self, filename='generator.png'):
        self.generator.summary()
        plot_model(self.generator, filename, show_shapes=True)

    def show_discriminator_model(self, filename='discriminator.png'):
        self.discriminator.summary()
        plot_model(self.discriminator, filename, show_shapes=True)

    def save_model(self, dir_name):
        make_directory(dir_name)

        save_model_to_json(self.generator, os.path.join(dir_name, 'generator_model.json'))
        self.generator.save_weights(os.path.join(dir_name, 'generator_weights.h5'))

        save_model_to_json(self.discriminator, os.path.join(dir_name, 'discriminator_model.json'))
        self.discriminator.save_weights(os.path.join(dir_name, 'discriminator_weights.h5'))


def plot_image(x):
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(5, 5)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(x):
        ax = fig.add_subplot(gs[i])
        ax.axis('off')
        ax.imshow(sample.reshape(28, 28), cmap='Greys_r')

    plt.show()
    plt.close(fig)


def masking_image(x):
    n_image = x.shape[0]
    height = x.shape[1]
    width = x.shape[2]

    for i in range(0, n_image):
        n_mask = np.random.randint(4, 8)

        for _ in range(0, n_mask):
            mask_length = np.random.randint(5, 8)
            height_point = np.random.randint(0, height - mask_length)
            width_point = np.random.randint(0, width - mask_length)

            x[i, height_point:height_point + mask_length, width_point:width_point + mask_length] = np.nan

    return x


if __name__ == '__main__':
    from tensorflow.python.keras.datasets import fashion_mnist

    (x_train, _), (x_test, _) = fashion_mnist.load_data()

    scaled_x_train = x_train / 255.0
    scaled_x_test = x_test / 255.0

    idx = np.random.permutation(scaled_x_train.shape[0])[:25]
    scaled_x_train_sample = scaled_x_train[idx, :]

    plot_image(scaled_x_train_sample)

    scaled_x_train_sample = masking_image(scaled_x_train_sample)

    plot_image(scaled_x_train_sample)

    # scaled_x_train = scaled_x_train.reshape(scaled_x_train.shape[0], scaled_x_train.shape[1] * scaled_x_train.shape[2])
    # scaled_x_test = scaled_x_test.reshape(scaled_x_test.shape[0], scaled_x_test.shape[1] * scaled_x_test.shape[2])
    #
    # gain_model = GAIN(input_shape=[scaled_x_train.shape[1]], alpha=50)
    # gain_model.fit(scaled_x_train, validation=scaled_x_test, log_dir="fashion_mnist")
