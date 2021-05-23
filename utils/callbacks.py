from keras.callbacks import Callback, LearningRateScheduler
import numpy as np
import settings.constants as constants
import matplotlib.pyplot as plt
import os


class CustomCallback(Callback):
    def __init__(self, run_folder, print_every_n_batches, initial_epoch, vae):
        self.epoch = initial_epoch
        self.run_folder = run_folder
        self.print_every_n_batches = print_every_n_batches
        self.vae = vae

    def on_batch_end(self, batch_index, logs={}):
        if batch_index % self.print_every_n_batches == 0:
            z_new = np.random.normal(size=(1, self.vae.z_dim))
            reconstructed = self.vae.decoder.predict(np.array(z_new))[0].squeeze()

            filepath = os.path.join(
                self.run_folder, constants.SAMPLE_RESULTS_FOLDER_NAME, "img_" + str(self.epoch).zfill(3) + "_" + str(batch_index) + ".jpg"
            )
            if len(reconstructed.shape) == 2:
                plt.imsave(filepath, reconstructed, cmap="gray_r")
            else:
                plt.imsave(filepath, reconstructed)

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch += 1


def step_decay_schedule(initial_lr, decay_factor=0.5, step_size=1):
    def schedule(epoch):
        new_lr = initial_lr * (decay_factor ** np.floor(epoch / step_size))

        return new_lr

    return LearningRateScheduler(schedule)
