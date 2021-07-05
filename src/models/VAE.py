from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout
from keras.models import Model
from keras import backend as K
from settings import constants
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from utils.callbacks import TrainingReferenceReconstructor, step_decay_schedule
import numpy as np
import os
import pickle
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


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
            use_batch_norm=False,
            use_dropout=False,
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
        :param use_batch_norm: Use batch normalization flag.
        :param use_dropout: Use dropout layers flag.
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

        self._build()

    def _build(self):
        """
        Create encoder/decoder model, apply constructor parameters into network creation.
        """
        # Encoder part.
        encoder_input = Input(shape=self.input_dim, name='encoder_input')

        x = encoder_input

        for i in range(self.n_layers_encoder):
            conv_layer = Conv2D(
                filters=self.encoder_conv_filters[i],
                kernel_size=self.encoder_conv_kernel_size[i],
                strides=self.encoder_conv_strides[i],
                padding="same",
                name="encoder_conv_" + str(i),
            )

            x = conv_layer(x)

            if self.use_batch_norm:
                x = BatchNormalization()(x)

            x = LeakyReLU()(x)

            if self.use_dropout:
                x = Dropout(rate=0.25)(x)

        # No batch size needed.
        shape_before_flattening = K.int_shape(x)[1:]

        x = Flatten()(x)
        self.mu = Dense(self.z_dim, name="mu")(x)
        self.log_var = Dense(self.z_dim, name="log_var")(x)

        self.encoder_mu_log_var = Model(encoder_input, (self.mu, self.log_var))

        def sampling(args):
            mu, log_var = args
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
            return mu + K.exp(log_var / 2) * epsilon

        encoder_output = Lambda(sampling, name="encoder_output")([self.mu, self.log_var])

        self.encoder = Model(encoder_input, encoder_output)

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
                x = Activation('sigmoid')(x)

        decoder_output = x

        self.decoder = Model(decoder_input, decoder_output)

        # Combined model.
        model_input = encoder_input
        # We need symbolic output to join both models.
        model_output = self.decoder(encoder_output)

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
            kl_loss = kl_divergence_loss(y_true, y_pred)
            return r_loss + kl_loss

        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss=total_loss, metrics=[reconstruction_loss, kl_divergence_loss])

    def save(self, folder):
        """
        Serialize model parameters using pickle and save it.
        :param folder: File to save parameters in.
        """
        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, constants.NETWORK_VISUALIZATION_FOLDER_NAME))
            os.makedirs(os.path.join(folder, constants.WEIGHTS_FOLDER_NAME))
            os.makedirs(os.path.join(folder, constants.SAMPLE_RESULTS_FOLDER_NAME))

        with open(os.path.join(folder, "params.pkl"), "wb") as f:
            pickle.dump([
                self.input_dim,
                self.encoder_conv_filters,
                self.encoder_conv_kernel_size,
                self.encoder_conv_strides,
                self.decoder_conv_t_filters,
                self.decoder_conv_t_kernel_size,
                self.decoder_conv_t_strides,
                self.z_dim,
                self.use_batch_norm,
                self.use_dropout,
            ], f)

        self.plot_model(folder)

    @staticmethod
    def load(model_class, folder):
        """
        Create model instance and apply input parameters by deserializing pickle file.
        :param model_class: Class of the model (this).
        :param folder: Source pkl file.
        :return: Constructed model.
        """
        with open(os.path.join(folder, "params.pkl"), "rb") as f:
            params = pickle.load(f)

        model = model_class(*params)
        model.load_weights(os.path.join(folder, "weights/weights.h5"))

        return model

    def load_weights(self, filepath):
        """
        Load and apply saved weights to model.
        :param filepath: File where weights are stored.
        """
        self.model.load_weights(filepath)

    def train(self, x_train, batch_size, epochs, run_folder, execute_on_nth_batch=100, initial_epoch=0, lr_decay=1):
        """
        Train the model with regular discrete data.
        :param x_train: Input train features.
        :param batch_size: Batch size.
        :param epochs: Epochs amount.
        :param run_folder: Run folder path.
        :param execute_on_nth_batch: Nth batch to execute custom callback on.
        :param initial_epoch: Initial epoch.
        :param lr_decay: Learning rate decay.
        """
        custom_callback = TrainingReferenceReconstructor(run_folder, execute_on_nth_batch, initial_epoch, self)
        lr_schedule = step_decay_schedule(initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)

        checkpoint1 = ModelCheckpoint(os.path.join(run_folder, "weights/weights.h5"), save_weights_only=True, verbose=1)
        callbacks_list = [checkpoint1, custom_callback, lr_schedule]

        self.model.fit(
            x_train,
            x_train,
            batch_size=batch_size,
            shuffle=True,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks_list,
        )

    def train_with_generator(self, data_flow, epochs, steps_per_epoch, run_folder, execute_on_nth_batch=100, initial_epoch=0, lr_decay=1):
        """
        Train the model with dataflow.
        :param data_flow: Input dataflow.
        :param epochs: Epochs amount.
        :param steps_per_epoch: Steps per epoch.
        :param run_folder: Run folder path.
        :param execute_on_nth_batch: Nth batch to execute custom callback on.
        :param initial_epoch: Initial epoch.
        :param lr_decay: Learning rate decay.
        """
        custom_callback = TrainingReferenceReconstructor(run_folder, execute_on_nth_batch, initial_epoch, self)
        lr_schedule = step_decay_schedule(initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)

        checkpoint1 = ModelCheckpoint(os.path.join(run_folder, "weights/weights.h5"), save_weights_only=True, verbose=1)
        callbacks_list = [checkpoint1, custom_callback, lr_schedule]

        self.model.save_weights(os.path.join(run_folder, "weights/weights.h5"))

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