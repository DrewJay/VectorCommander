import os
from glob import glob
import numpy as np
from models.VAE import VariationalAutoencoder
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


# Create folder run/vae/${run_id}_${data_name}/viz, images, weights
section = "vae"
run_id = "0001"
data_name = "faces"
RUN_FOLDER = "run/%s/" % section
RUN_FOLDER += "_".join([run_id, data_name])  # Array means join from left and right.

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, "viz"))
    os.mkdir(os.path.join(RUN_FOLDER, "images"))
    os.mkdir(os.path.join(RUN_FOLDER, "weights"))

mode = "build"
DATA_FOLDER = "./data/train/"

INPUT_DIM = (128, 128, 3)
BATCH_SIZE = 32
filenames = np.array(glob(os.path.join(DATA_FOLDER, "*/*.jpg")))
NUM_IMAGES = len(filenames)

data_gen = ImageDataGenerator(rescale=1./255)
data_flow = data_gen.flow_from_directory(
    DATA_FOLDER,
    target_size=INPUT_DIM[:2],
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode="input",  # Data = [images_batch_1, images_batch_2, images_batch_3, ...]
    subset="training",
)

vae = VariationalAutoencoder(
    input_dim=INPUT_DIM,
    encoder_conv_filters=[32, 64, 64, 64],
    encoder_conv_kernel_size=[3, 3, 3, 3],
    encoder_conv_strides=[2, 2, 2, 2],
    decoder_conv_t_filters=[64, 64, 32, 3],
    decoder_conv_t_kernel_size=[3, 3, 3, 3],
    decoder_conv_t_strides=[2, 2, 2, 2],
    z_dim=200,
    use_batch_norm=True,
    use_dropout=True,
)

if mode == "build":
    vae.save(RUN_FOLDER)
else:
    vae.load_weights(os.path.join(RUN_FOLDER, "weights/weights.h5"))

LEARNING_RATE = 0.0005
R_LOSS_FACTOR = 10000
EPOCHS = 10
PRINT_EVERY_N_BATCHES = 100
INITIAL_EPOCH = 0

vae.compile(LEARNING_RATE, R_LOSS_FACTOR)

vae.train_with_generator(
    data_flow,
    epochs=EPOCHS,
    steps_per_epoch=NUM_IMAGES / BATCH_SIZE,
    run_folder=RUN_FOLDER,
    print_every_n_batches=PRINT_EVERY_N_BATCHES,
    initial_epoch=INITIAL_EPOCH,
)
