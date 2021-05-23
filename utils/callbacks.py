from keras.callbacks import Callback, LearningRateScheduler
import numpy as np
import settings.constants as constants
import matplotlib.pyplot as plt
import os


class TrainingReferenceReconstructor(Callback):
    """
    Custom callback generating random samples using decoder during training.
    """
    def __init__(self, run_folder, execute_on_nth_batch, initial_epoch, vae):
        """
        Class constructor.
        :param run_folder: Run folder path.
        :param execute_on_nth_batch: Execute callback on every nth batch.
        :param initial_epoch: Initial epoch index.
        :param vae: VAE model reference.
        """
        self.epoch = initial_epoch
        self.run_folder = run_folder
        self.execute_on_nth_batch = execute_on_nth_batch
        self.vae = vae

    def on_batch_end(self, batch_index, logs={}):
        """
        Generate reconstructed images on batch end.
        :param batch_index: Current batch index.
        :param logs: Metrics results logs.
        """
        if batch_index % self.execute_on_nth_batch == 0:
            z_new = np.random.normal(size=(1, self.vae.z_dim))
            reconstructed = self.vae.decoder.predict(np.array(z_new))[0].squeeze()

            filepath = os.path.join(
                self.run_folder, constants.SAMPLE_RESULTS_FOLDER_NAME, "img_" + str(self.epoch).zfill(3) + "_" + str(batch_index) + ".jpg"
            )
            # If no channels, make it grayscale.
            if len(reconstructed.shape) == 2:
                plt.imsave(filepath, reconstructed, cmap="gray_r")
            else:
                plt.imsave(filepath, reconstructed)

    def on_epoch_begin(self, epoch, logs={}):
        """
        Increment local epoch counter on epoch begin.
        :param epoch: Current epoch.
        :param logs: Metrics results logs.
        """
        self.epoch = epoch


def step_decay_schedule(initial_lr, decay_factor=0.5, step_size=1):
    """
    Create Keras LearningRateScheduler to control learning rate hyper parameter during training.
    :param initial_lr: Initial learning rate.
    :param decay_factor: Decay factor.
    :param step_size: Step Size.
    :return: LearningRateScheduler instance.
    """
    def schedule(epoch):
        """
        Callback calculating new learning rate.
        :param epoch: Current epoch.
        :return: New Learning rate.
        """
        new_lr = initial_lr * (decay_factor ** np.floor(epoch / step_size))
        return new_lr

    return LearningRateScheduler(schedule)
