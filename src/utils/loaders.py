import pickle
import os
from keras.preprocessing.image import ImageDataGenerator


class ImageLabelLoader:
    """
    Creates image loader using Keras ImageDataGenerator instance.
    """
    def __init__(self, image_folder, target_size):
        """
        Class constructor.
        :param image_folder: Path to main dataset.
        :param target_size: Target image size.
        """
        self.image_folder = image_folder
        self.target_size = target_size

    def build(self, csv_data, batch_size, label=None):
        """
        Create flow_from_dataframe instance for either labeled or unlabeled data.
        :param csv_data: Pandas dataframe (CSV).
        :param batch_size: Batch size.
        :param label: Label of interest.
        :return: flow_from_dataframe instance.
        """
        data_gen = ImageDataGenerator(rescale=1. / 255)
        if label:
            # Returns amount_of_images/batch_size tuples where (image_batch, y_col).
            # If y_col = ["a", "b"] then [[0, 1], [1, 0], ...x batch_size] a.k.a (batch_size, 2).
            # If y_col = "a" then [0, 1, ...x batch_size] a.k.a (batch_size).
            data_flow = data_gen.flow_from_dataframe(
                csv_data,  # Pandas CSV containing data information.
                self.image_folder,  # Where to look for images.
                x_col="Image Index",  # Image names.
                y_col=label,  # Name of column(s) in CSV whose values will be placed in returned tuples as 2nd value.
                target_size=self.target_size,  # No channels, only spatial dimensions.
                class_mode="raw",  # y_col in resulting tuples will be simple numpy array.
                batch_size=batch_size,
                shuffle=True,
            )
        else:
            # "input" class mode means that second item in the tuple
            # is the same image batch as first item a.k.a (batch_size, w, h, c).
            data_flow = data_gen.flow_from_dataframe(
                csv_data,
                self.image_folder,
                x_col="Image Index",
                target_size=self.target_size,
                class_mode="input",
                batch_size=batch_size,
                shuffle=True,
            )

        return data_flow
