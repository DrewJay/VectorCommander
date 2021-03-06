# Folder generation settings.
WRAPPER_FOLDER_NAME = "vae"
RUN_ID = "1"
RESULTS_FOLDER_NAME = "output"
RUN_FOLDER_NAME = ("run/%s/" % WRAPPER_FOLDER_NAME) + "_".join([RUN_ID, RESULTS_FOLDER_NAME])
MODE = "build"
DATA_FOLDER_NAME = "data/train/"
IMAGE_FOLDER = "data/train/images/"
VECTOR_FOLDER = "run/analysis/label_vectors/"
NETWORK_VISUALIZATION_FOLDER_NAME = "architecture"
SAMPLE_RESULTS_FOLDER_NAME = "generatedSamples"

CSV_NAME = "Data_Entry_2017_v2020.csv"
CSV_X_COL = "Image Index"
DATA_EXTENSION = "png"

# Model metadata.
LEARNING_RATE = 0.001
Z_DIM = 200
GAMMA = 1
TARGET_CAPACITY = 0
RECONSTRUCTION_LOSS_FACTOR = 10000

# Training metadata.
INPUT_DIM = (128, 128, 3)
BATCH_SIZE = 32
EPOCHS = 100
EXEC_ON_NTH_BATCH = 9
EXEC_ON_NTH_EPOCH = 2
INITIAL_EPOCH = 0

# Analysis metadata.
ANALYSIS_BATCH_SIZE = 300

# Various config.
DISCRIMINATIVE = False
USE_DELTA = True
PLOT_TRAINING_LOSS = False
