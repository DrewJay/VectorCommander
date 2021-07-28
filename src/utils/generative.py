import settings.constants as constants
import os


def gen_dirs():
    """
        Generate run directory and necessary subdirectories if missing.
    """
    if not os.path.exists(constants.RUN_FOLDER_NAME):
        os.makedirs(constants.RUN_FOLDER_NAME)
        os.mkdir(os.path.join(constants.RUN_FOLDER_NAME, constants.NETWORK_VISUALIZATION_FOLDER_NAME))
        os.mkdir(os.path.join(constants.RUN_FOLDER_NAME, constants.SAMPLE_RESULTS_FOLDER_NAME))
        os.mkdir(os.path.join(constants.RUN_FOLDER_NAME, constants.WEIGHTS_FOLDER_NAME))

    if not os.path.exists(constants.VECTOR_FOLDER):
        os.makedirs(constants.VECTOR_FOLDER)
