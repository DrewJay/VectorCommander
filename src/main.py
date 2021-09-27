import os
from glob import glob
import numpy as np
import settings.constants as constants
import utils.generative as generative
from models.VAE import VariationalAutoencoder
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.framework.ops import disable_eager_execution
from utils.callbacks import TrainingReferenceReconstructor, step_decay_schedule
disable_eager_execution()

generative.gen_dirs()

filenames = np.array(glob(os.path.join(constants.DATA_FOLDER_NAME, "*/*." + constants.DATA_EXTENSION)))
IMAGES_AMOUNT = len(filenames)

data_gen = ImageDataGenerator(rescale=1./255)
# This method loads raw images without any labels in subdirectories of DATA_FOLDER_NAME directory.
data_flow = data_gen.flow_from_directory(
    constants.DATA_FOLDER_NAME,
    target_size=constants.INPUT_DIM[:2],
    batch_size=constants.BATCH_SIZE,
    shuffle=True,
    class_mode="input",
    subset="training",
)

print("Discriminative mode: " + str(constants.DISCRIMINATIVE))

vae = VariationalAutoencoder(
    input_dim=constants.INPUT_DIM,
    encoder_conv_filters=[32, 64, 64, 128],
    encoder_conv_kernel_size=[3, 3, 3, 3],
    encoder_conv_strides=[2, 2, 2, 2],
    decoder_conv_t_filters=[64, 64, 32, 3],
    decoder_conv_t_kernel_size=[3, 3, 3, 3],
    decoder_conv_t_strides=[2, 2, 2, 2],
    z_dim=constants.Z_DIM,
    use_batch_norm=True,
    use_dropout=True,
    discriminative=constants.DISCRIMINATIVE
)

# Compile the model.
vae.compile(constants.LEARNING_RATE, constants.RECONSTRUCTION_LOSS_FACTOR)

# Vanilla VAE training.
if not vae.discriminative:
    vae.train_with_generator(
        data_flow,
        epochs=constants.EPOCHS,
        steps_per_epoch=IMAGES_AMOUNT / constants.BATCH_SIZE,
        run_folder=constants.RUN_FOLDER_NAME,
        execute_on=(constants.EXEC_ON_NTH_BATCH, constants.EXEC_ON_NTH_EPOCH),
        initial_epoch=constants.INITIAL_EPOCH,
    )

# AAE training.
else:
    valid_labels = np.ones((constants.BATCH_SIZE, 1))
    fake_labels = np.zeros((constants.BATCH_SIZE, 1))
    callback_invoker = TrainingReferenceReconstructor(constants.RUN_FOLDER_NAME, 0, 0, vae)

    for epoch in range(constants.EPOCHS):
        images = data_flow.next()
        callback_invoker.epoch = epoch
        batch_count = 1

        while np.shape(images)[1] is constants.BATCH_SIZE:
            image = images[0]

            latent_fake = vae.encoder.predict(image)
            latent_real = np.random.normal(size=(constants.BATCH_SIZE, constants.Z_DIM))

            # Train discriminator on the batch.
            disc_loss_real = vae.discriminator.train_on_batch(latent_real, valid_labels)
            disc_loss_fake = vae.discriminator.train_on_batch(latent_fake, fake_labels)
            disc_loss = 0.5 * np.add(disc_loss_real, disc_loss_fake)

            # Train combined model on the batch.
            aae_loss = vae.model.train_on_batch(image, [image, valid_labels])

            # Plot the progress.
            print("Epoch %d: [Discriminator loss: %f, accuracy: %.2f%%] [Generator loss: %f, mse: %f]" % (epoch, disc_loss[0], 100 * disc_loss[1], aae_loss[0], aae_loss[1]))

            images = data_flow.next()
            batch_count += 1

        callback_invoker.execute_on_nth_batch = batch_count
        callback_invoker.on_batch_end(batch_count)

# Save the model.
vae.model.save(os.path.join(constants.RUN_FOLDER_NAME, "model"))
