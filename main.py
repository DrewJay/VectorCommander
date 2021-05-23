import os
from glob import glob
import numpy as np
import settings.constants as constants
import utils.generative as generative
from models.VAE import VariationalAutoencoder
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

generative.gen_dirs()

filenames = np.array(glob(os.path.join(constants.DATA_FOLDER_NAME, "*/*.jpg")))
IMAGES_AMOUNT = len(filenames)

data_gen = ImageDataGenerator(rescale=1./255)
# This method loads raw images without any labels in subdirectories of DATA_FOLDER_NAME directory.
data_flow = data_gen.flow_from_directory(
    constants.DATA_FOLDER_NAME,
    target_size=constants.INPUT_DIM[:2],
    batch_size=constants.BATCH_SIZE,
    shuffle=True,
    class_mode="input",  # Data = [images_batch_1, images_batch_2, images_batch_3, ...].
    subset="training",
)

vae = VariationalAutoencoder(
    input_dim=constants.INPUT_DIM,
    encoder_conv_filters=[32, 64, 64, 64],
    encoder_conv_kernel_size=[3, 3, 3, 3],
    encoder_conv_strides=[2, 2, 2, 2],
    decoder_conv_t_filters=[64, 64, 32, 3],
    decoder_conv_t_kernel_size=[3, 3, 3, 3],
    decoder_conv_t_strides=[2, 2, 2, 2],
    z_dim=constants.Z_DIM,
    use_batch_norm=True,
    use_dropout=True,
)

if constants.MODE == "build":
    vae.save(constants.RUN_FOLDER_NAME)
else:
    vae.load_weights(os.path.join(constants.RUN_FOLDER_NAME, "weights/weights.h5"))

vae.compile(constants.LEARNING_RATE, constants.RECONSTRUCTION_LOSS_FACTOR)

vae.train_with_generator(
    data_flow,
    epochs=constants.EPOCHS,
    steps_per_epoch=IMAGES_AMOUNT / constants.BATCH_SIZE,
    run_folder=constants.RUN_FOLDER_NAME,
    execute_on_nth_batch=constants.EXEC_ON_NTH_BATCH,
    initial_epoch=constants.INITIAL_EPOCH,
)
