import tensorflow
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.python.framework.ops import disable_eager_execution
from keras.callbacks import Callback, LearningRateScheduler
import os
import gc

disable_eager_execution()

DATA_FOLDER_NAME = "G:/celeba/img_align_celeba"
CSV_FILE = "data/train/list_attr_celeba.csv"
image_size = 128
BATCH_SIZE = 64
latent_space = 200

data_gen = ImageDataGenerator(rescale=1./255)
data_flow = data_gen.flow_from_directory(
    DATA_FOLDER_NAME,
    target_size=(image_size, image_size),
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode="input",
    subset="training",
)


def build_generator(latent_space_size):
    model_input = layers.Input(shape=latent_space_size)

    x = layers.Dense(
        units=(8 * 8 * 5),
        input_shape=(latent_space_size,),
    )(model_input)
    x = layers.Reshape(target_shape=(8, 8, 5))(x)
    x = layers.Conv2DTranspose(
        filters=128,
        kernel_size=4,
        strides=2,
        padding="same",
    )(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2DTranspose(
        filters=256,
        kernel_size=4,
        strides=2,
        padding="same",
    )(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2DTranspose(
        filters=512,
        kernel_size=4,
        strides=2,
        padding="same",
    )(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2DTranspose(
        filters=3,
        kernel_size=4,
        strides=2,
        padding="same",
        activation="sigmoid",
    )(x)

    return Model(inputs=[model_input], outputs=[x])


def build_discriminator():
    model_input = layers.Input(shape=(image_size, image_size, 3))

    x = layers.Conv2D(
        filters=64,
        kernel_size=4,
        padding="same",
        strides=2,
        input_shape=(image_size, image_size, 1),
    )(model_input)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(
        filters=128,
        kernel_size=4,
        padding="same",
        strides=2,
    )(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(
        filters=128,
        kernel_size=4,
        padding="same",
        strides=2,
    )(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(rate=0.2)(x)

    realness = layers.Dense(units=1, activation="sigmoid")(x)

    return Model(inputs=[model_input], outputs=[realness])


def build_combined_model(latent_space_size, generator_model, discriminator_model):
    model_input = layers.Input(shape=latent_space_size)
    fake_image = generator_model(model_input)

    discriminator_model.trainable = False
    fakeness = discriminator_model(fake_image)

    combined_model = Model(inputs=[model_input], outputs=[fakeness])
    combined_model.compile(loss=["binary_crossentropy"], optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
    return combined_model


generator = build_generator(latent_space)
discriminator = build_discriminator()
discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="binary_crossentropy")
combined = build_combined_model(latent_space, generator, discriminator)


def train_discriminator_one_step(xs, latent_space_size, batch_size):
    z_vectors = tf.random.uniform([batch_size, latent_space_size], -1, 1)

    generated_images = generator.predict([z_vectors], steps=1, batch_size=1)
    x = tf.concat([xs, generated_images], axis=0)
    y = tf.concat([tf.multiply(tf.ones([batch_size, 1]), 1), tf.zeros([batch_size, 1])], axis=0)

    loss = discriminator.train_on_batch([x], [y])
    return loss


def train_combined_model_one_step(latent_space_size, batch_size):
    z_vectors = tf.random.uniform([batch_size, latent_space_size], -1, 1)
    trick = tf.multiply(tf.ones([batch_size, 1]), 1)

    loss = combined.train_on_batch([z_vectors], [trick])
    return loss


class TrainingReferenceReconstructor(Callback):
    def __init__(
            self,
            generator_model,
            latent_space_size,
    ):
        self.generator_model = generator_model
        self.latent_space_size = latent_space_size
        self.epoch = 0

    def on_epoch_begin(self, epoch, logs={}):
        """
        Increment local epoch counter on epoch begin.
        :param epoch: Current epoch.
        :param logs: Metrics results logs.
        """
        self.epoch = epoch

    def on_batch_end(self, batch_index, logs={}):
        """
        Generate reconstructed images on batch end.
        :param batch_index: Current batch index.
        :param logs: Metrics results logs.
        """
        z_vectors = tf.random.uniform((1, self.latent_space_size), -1, 1)

        # Remove batch dim because it has only 1 item inside.
        reconstructed = self.generator_model.predict([z_vectors], steps=1)[0].squeeze()

        filepath = os.path.join(
            "run/img", "img_" + str(self.epoch).zfill(3) + "_" + str(batch_index) + ".jpg"
        )

        img = keras.preprocessing.image.array_to_img(reconstructed * 255)
        img.save(filepath)


cb = TrainingReferenceReconstructor(generator, latent_space)
epochs = 30
for epoch in range(epochs):
    images = data_flow.next()
    batch_count = 1

    while np.shape(images)[1] is BATCH_SIZE:
        image_batch = images[0]

        image_batch_half = image_batch[:len(image_batch)//2]

        disc_loss = train_discriminator_one_step(image_batch_half, latent_space, int(BATCH_SIZE/2))
        combined_loss = train_combined_model_one_step(latent_space, BATCH_SIZE)

        images = data_flow.next()

        cb.on_batch_end(batch_count)

        print("Epoch: " + str(cb.epoch) + ", batch: " + str(batch_count))
        batch_count += 1

    cb.epoch += 1
