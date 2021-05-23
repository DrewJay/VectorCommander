import pickle
import os
from keras.preprocessing.image import ImageDataGenerator


class ImageLabelLoader:
    def __init__(self, image_folder, target_size):
        self.image_folder = image_folder
        self.target_size = target_size

    def build(self, att, batch_size, label=None):
        data_gen = ImageDataGenerator(rescale=1. / 255)
        if label:
            # Returns amount_of_images/batch_size tuples where (image_batch, y_col).
            # If y_col = ["a", "b"] then [[[0, 1], [1, 0]] x batch_size].
            # If y_col = "a" then [0, 1, ...x batch_size].
            # If y_col = None then it's not tuple.
            data_flow = data_gen.flow_from_dataframe(
                att,  # Pandas CSV containing data information.
                self.image_folder,  # Where to look for images.
                x_col="image_id",  # Image names.
                y_col=label,  # Name of column(s) in CSV whose values will be placed in returned tuples as 2nd value.
                target_size=self.target_size,  # No channels, only spatial dimensions.
                class_mode="raw",  # y_col in resulting tuples will be simple numpy array.
                batch_size=batch_size,
                shuffle=True,
            )
        else:
            data_flow = data_gen.flow_from_dataframe(
                att,
                self.image_folder,
                x_col="image_id",
                target_size=self.target_size,
                class_mode="input",
                batch_size=batch_size,
                shuffle=True,
            )

        return data_flow


def load_model(model_class, folder):
    with open(os.path.join(folder, "params.pkl"), "rb") as f:
        params = pickle.load(f)

    model = model_class(*params)

    model.load_weights(os.path.join(folder, "weights/weights.h5"))

    return model
