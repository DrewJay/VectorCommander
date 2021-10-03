import os
from glob import glob
import numpy as np
import settings.constants as constants
import utils.generative as generative
from models.VAE import VariationalAutoencoder
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.framework.ops import disable_eager_execution
from utils.callbacks import TrainingReferenceReconstructor
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

print("Discriminative mode: " + str(constants.DISCRIMINATIVE) + ".")
print("Delta mode: " + str(constants.USE_DELTA) + ".")

vae = VariationalAutoencoder(
    input_dim=constants.INPUT_DIM,
    encoder_conv_filters=[32, 32, 64, 128, 256],
    encoder_conv_kernel_size=[3, 3, 3, 3, 3],
    encoder_conv_strides=[2, 1, 2, 2, 2],
    decoder_conv_t_filters=[128, 64, 32, 32, 3],
    decoder_conv_t_kernel_size=[3, 3, 3, 3, 3],
    decoder_conv_t_strides=[2, 2, 2, 1, 2],
    z_dim=constants.Z_DIM,
    dense_units=[500, 250, 125, 64, 32],
    use_batch_norm=True,
    use_dropout=True,
    discriminative=constants.DISCRIMINATIVE,
    gamma=constants.GAMMA,
    capacity=constants.CAPACITY,
    use_delta=constants.USE_DELTA
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
    valid_labels = np.ones((constants.BATCH_SIZE, 1), dtype=int)
    fake_labels = np.zeros((constants.BATCH_SIZE, 1), dtype=int)

    callback_invoker = TrainingReferenceReconstructor(
        run_folder=constants.RUN_FOLDER_NAME,
        execute_on=(constants.EXEC_ON_NTH_BATCH, constants.EXEC_ON_NTH_EPOCH),
        initial_epoch=constants.INITIAL_EPOCH,
        vae=vae,
        plot_training_loss=constants.PLOT_TRAINING_LOSS
    )

    for epoch in range(constants.EPOCHS):
        images = data_flow.next()
        batch_count = 1

        while np.shape(images)[1] is constants.BATCH_SIZE:
            image = images[0]

            latent_fake = vae.encoder.predict(image)
            latent_real = np.random.normal(size=(constants.BATCH_SIZE, constants.Z_DIM))
            # Train discriminator on the batch.
            disc_loss = vae.discriminator.train_on_batch(x=np.concatenate((latent_real, latent_fake)), y=np.concatenate((valid_labels, fake_labels)))

            # Train combined model on the batch.
            aae_loss = vae.model.train_on_batch(image, [image, valid_labels])

            # Plot the progress.
            print("Epoch %d: [Discriminator loss: %f, Accuracy: %.2f%%] [MSE: %f, Validity Loss: %f]" % (epoch, disc_loss[0], 100 * disc_loss[1], aae_loss[0], aae_loss[1]))

            images = data_flow.next()

            callback_invoker.on_batch_end(batch_count)
            batch_count += 1

        callback_invoker.on_epoch_end(epoch, {"disc_loss": disc_loss[0], "reconstruction_loss": aae_loss[0], "validity_loss": aae_loss[1]})
        callback_invoker.epoch += 1

# Save the model.
vae.model.save(os.path.join(constants.RUN_FOLDER_NAME, "model"))
