from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, Layer, Add
from keras.initializers import glorot_normal
from keras.models import Model
from keras import backend as K
from settings import constants
from keras.optimizers import Adam
from keras.utils import plot_model
from utils.callbacks import TrainingReferenceReconstructor, step_decay_schedule
import numpy as np
import os
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Sampling(Layer):
    """
    Sampling layer definition.
    """
    def __init__(self, initializer=glorot_normal, **kwargs):
        """
        Constructor.
        :param initializer: Weights initializer.
        :param kwargs: Just kwargs.
        """
        super(Sampling, self).__init__(**kwargs)
        self.initializer = initializer
        self._name = "encoder_output"

    def call(self, args):
        """
        Samples from random distribution.
        :param args: Layer inputs.
        :return: Generated random distribution.
        """
        mu, log_var, delta = args
        # We cannot directly sample from distribution using given mean and variance, because backpropagation would be stochastic
        # with increased variance and thus less accurate. Solution to this issue is sampling for known distribution, multiplying it by
        # standard deviation and adding mean tensors to it.
        epsilon = delta * K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
        # Square root of variance.
        return mu + K.exp(log_var / 2) * epsilon


class VariationalAutoencoder:
    """
    VAE model class.
    """
    def __init__(
            self,
            input_dim,
            encoder_conv_filters,
            encoder_conv_kernel_size,
            encoder_conv_strides,
            decoder_conv_t_filters,
            decoder_conv_t_kernel_size,
            decoder_conv_t_strides,
            z_dim,
            dense_units,
            gamma=1,
            capacity=0,
            use_batch_norm=False,
            use_dropout=False,
            discriminative=False,
    ):
        """
        Class constructor.
        :param input_dim: Input data dimensions.
        :param encoder_conv_filters: Encoder convolution filters.
        :param encoder_conv_kernel_size: Encoder convolution kernel size.
        :param encoder_conv_strides: Encoder convolution strides.
        :param decoder_conv_t_filters: Decoder convolution filters.
        :param decoder_conv_t_kernel_size: Decoder convolution kernel size.
        :param decoder_conv_t_strides: Decoder convolution strides.
        :param z_dim: Latent vector shape.
        :param dense_units: Array of units of dense layers to use in discriminator.
        :param use_batch_norm: Use batch normalization flag.
        :param use_dropout: Use dropout layers flag.
        :param discriminative: Use discriminator instead of KL divergence.
        """
        self.name = "variational_autoencoder"

        self.input_dim = input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides
        self.z_dim = z_dim
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.n_layers_encoder = len(encoder_conv_filters)
        self.n_layers_decoder = len(decoder_conv_t_filters)
        self.discriminative = discriminative
        self.dense_units = dense_units
        self.gamma = gamma
        self.capacity = capacity

        self._build()

    def create_residual_block(self, x, index):
        """
        Instead of using classical convolutional block, generate
        residual one.
        :param x: Input tensor.
        :param index: Block index.
        :return: Built residual block.
        """
        y = Conv2D(
            filters=self.encoder_conv_filters[index],
            kernel_size=self.encoder_conv_kernel_size[index],
            strides=self.encoder_conv_strides[index],
            padding="same",
            name="encoder_conv_" + str(index),
        )(x)
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)

        y = Conv2D(
            filters=self.encoder_conv_filters[index],
            kernel_size=self.encoder_conv_kernel_size[index],
            strides=1,
            padding="same",
            name="encoder_conv_" + str(index) + "_2",
        )(y)

        # If y was downsampled, x has to be too, so we can add it.
        if self.encoder_conv_strides[index] > 1:
            x = Conv2D(
                kernel_size=1,
                strides=self.encoder_conv_strides[index],
                filters=self.encoder_conv_filters[index],
                padding="same"
            )(x)

        y = BatchNormalization()(y)
        out = Add()([x, y])
        out = LeakyReLU()(out)

        if self.use_dropout:
            out = Dropout(rate=0.25)(out)

        return out

    def _build(self):
        """
        Create encoder/decoder model, apply constructor parameters into network creation.
        """
        # Optional discriminator part.
        if self.discriminative:
            discriminator_input = Input(shape=self.z_dim)
            x = discriminator_input

            for units in self.dense_units:
                x = Dense(units)(x)
                x = LeakyReLU(alpha=0.2)(x)

            x = Dense(1, activation="sigmoid")(x)

            self.discriminator = Model(discriminator_input, x)
            self.discriminator._name = "discriminator"

        # Encoder part.
        encoder_input = Input(shape=self.input_dim, name='encoder_input')

        x = encoder_input

        for i in range(self.n_layers_encoder):
            x = self.create_residual_block(x=x, index=i)

        # No batch size needed.
        shape_before_flattening = K.int_shape(x)[1:]

        x = Flatten()(x)
        self.mu = Dense(self.z_dim, name="mu")(x)
        self.log_var = Dense(self.z_dim, name="log_var")(x)
        self.delta = Dense(1, name="delta")(x)

        self.encoder_mu_log_var = Model(encoder_input, (self.mu, self.log_var, self.delta))

        sampling = Sampling()([self.mu, self.log_var, self.delta])

        self.encoder = Model(encoder_input, sampling)
        self.encoder._name = "encoder"

        encoder_output = self.encoder(encoder_input)

        # Decoder part.
        decoder_input = Input(shape=(self.z_dim,), name="decoder_input")

        # Before we can reshape to for instance (32, 32, 3) data needs to be in shape (32 x 32 x 3).
        x = Dense(np.prod(shape_before_flattening))(decoder_input)
        x = Reshape(shape_before_flattening)(x)

        for i in range(self.n_layers_decoder):
            conv_t_layer = Conv2DTranspose(
                filters=self.decoder_conv_t_filters[i],
                kernel_size=self.decoder_conv_t_kernel_size[i],
                strides=self.decoder_conv_t_strides[i],
                padding="same",
                name="decoder_conv_t_" + str(i),
            )

            x = conv_t_layer(x)

            if i < self.n_layers_decoder - 1:
                if self.use_batch_norm:
                    x = BatchNormalization()(x)

                x = LeakyReLU()(x)

                if self.use_dropout:
                    x = Dropout(rate=0.25)(x)
            else:
                x = Activation('sigmoid', name="sigmoid")(x)

        decoder_output = x

        self.decoder = Model(decoder_input, decoder_output)
        self.decoder._name = "decoder"

        # Combined model.
        model_input = encoder_input
        # We need symbolic output to join both models.
        model_output = self.decoder(encoder_output)

        # Create discriminative combined model.
        if self.discriminative:
            validity = self.discriminator(encoder_output)
            # Not trainable during combined model training.
            self.model = Model(model_input, [model_output, validity])
        else:
            self.model = Model(model_input, model_output)

    def compile(self, learning_rate, r_loss_factor):
        """
        Compile the model.
        :param learning_rate: Learning rate.
        :param r_loss_factor: Reconstruction loss factor.
        """
        self.learning_rate = learning_rate

        # Reconstruction loss.
        def reconstruction_loss(y_true, y_pred):
            # Shape of data is [BATCH_SIZE, IMG_SIZE, IMG_SIZE, CHANNELS].
            # Calculates squared difference between two images and average pixel value for every batch.
            r_loss = K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])
            return r_loss_factor * r_loss

        # KL divergence loss.
        def kl_divergence_loss(y_true, y_pred):
            kl_loss = -0.5 * K.sum(1 + self.log_var - K.square(self.mu) - K.exp(self.log_var), axis=1)
            return kl_loss

        # Combined loss.
        def total_loss(y_true, y_pred):
            r_loss = reconstruction_loss(y_true, y_pred)
            kl_loss = self.gamma * K.abs(kl_divergence_loss(y_true, y_pred) - self.capacity)
            return r_loss + kl_loss

        loss = ["mse", "binary_crossentropy"] if self.discriminative else total_loss
        metrics = None if self.discriminative else [reconstruction_loss, kl_divergence_loss]

        # Compile discriminator if present.
        if self.discriminative:
            self.discriminator.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate), metrics=["accuracy"])
            self.discriminator.trainable = False

        # Compile the main model.
        self.model.compile(optimizer=Adam(lr=learning_rate), loss=loss, metrics=metrics)

    def train(self, x_train, batch_size, epochs, run_folder, execute_on=(100, 1), initial_epoch=0, lr_decay=1):
        """
        Train the model with regular discrete data.
        :param x_train: Input train features.
        :param batch_size: Batch size.
        :param epochs: Epochs amount.
        :param run_folder: Run folder path.
        :param execute_on: Nth batch/epoch to execute custom callback on.
        :param initial_epoch: Initial epoch.
        :param lr_decay: Learning rate decay.
        """
        custom_callback = TrainingReferenceReconstructor(
            run_folder=run_folder,
            execute_on=execute_on,
            initial_epoch=initial_epoch,
            vae=self,
            plot_training_loss=constants.PLOT_TRAINING_LOSS
        )
        lr_schedule = step_decay_schedule(initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)

        callbacks_list = [custom_callback, lr_schedule]

        self.model.fit(
            x_train,
            x_train,
            batch_size=batch_size,
            shuffle=True,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks_list,
        )

    def train_with_generator(self, data_flow, epochs, steps_per_epoch, run_folder, execute_on=(100, 1), initial_epoch=0, lr_decay=1):
        """
        Train the model with dataflow.
        :param data_flow: Input dataflow.
        :param epochs: Epochs amount.
        :param steps_per_epoch: Steps per epoch.
        :param run_folder: Run folder path.
        :param execute_on: Nth batch/epoch to execute custom callback on.
        :param initial_epoch: Initial epoch.
        :param lr_decay: Learning rate decay.
        """
        custom_callback = TrainingReferenceReconstructor(
            run_folder=run_folder,
            execute_on=execute_on,
            initial_epoch=initial_epoch,
            vae=self,
            plot_training_loss=constants.PLOT_TRAINING_LOSS
        )
        lr_schedule = step_decay_schedule(initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)

        callbacks_list = [custom_callback, lr_schedule]

        self.model.fit(
            data_flow,
            shuffle=True,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks_list,
            steps_per_epoch=steps_per_epoch,
        )

    def plot_model(self, run_folder):
        """
        Generate visual graph of this model.
        :param run_folder: Run folder path.
        """
        plot_model(
            self.model,
            to_file=os.path.join(run_folder, os.path.join(
                constants.NETWORK_VISUALIZATION_FOLDER_NAME, "model.png")
            ),
            show_shapes=True,
            show_layer_names=True
        )
        plot_model(
            self.encoder,
            to_file=os.path.join(
                run_folder, os.path.join(constants.NETWORK_VISUALIZATION_FOLDER_NAME, "encoder.png")
            ),
            show_shapes=True,
            show_layer_names=True
        )
        plot_model(
            self.decoder,
            to_file=os.path.join(run_folder, os.path.join(
                constants.NETWORK_VISUALIZATION_FOLDER_NAME, "decoder.png")
            ),
            show_shapes=True,
            show_layer_names=True
        )
