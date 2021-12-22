import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import norm
import settings.constants as constants
import pandas as pd
import argparse
from os import path
from utils.loaders import ImageLabelLoader
import keras

att = pd.read_csv(os.path.join(constants.DATA_FOLDER_NAME, constants.CSV_NAME))

# Exclude channels dim.
imageLoader = ImageLabelLoader(constants.IMAGE_FOLDER, constants.INPUT_DIM[:2], x_col=constants.CSV_X_COL)
vae = keras.models.load_model(os.path.join(constants.RUN_FOLDER_NAME, "model"), compile=False)

encoder = vae.get_layer("encoder")
decoder = vae.get_layer("decoder")

data_flow_unlabeled = imageLoader.build(csv_data=att, batch_size=10)


def show_distributions():
    """
    Show plot which compares random images encoded into latent vectors
    to standard normal distribution.
    """
    # 20 batches of size 10 = 200 images.
    z_test = encoder.predict_generator(data_flow_unlabeled, steps=20, verbose=1)
    x = np.linspace(-3, 3, 100)
    fig = plt.figure(figsize=(20, 20))
    fig.subplots_adjust(hspace=0.6, wspace=0.4)

    for i in range(50):
        # 5 x 10 grid.
        vector_display = fig.add_subplot(5, 10, i + 1)
        # This is not the right way to do it. Multivariate gaussian has to have independent members.
        # Therefore the right way is z_test[:, i].
        vector_display.hist(z_test[i], density=True, bins=20)
        vector_display.text(0.5, -0.35, "Vector " + str(i), c="red", fontsize=10, ha="center", transform=vector_display.transAxes)
        # pdf returns ys that represent bell curve itself in order, random_normal returns ys that follow
        # particular distribution without order.
        vector_display.plot(x, norm.pdf(x))
    plt.show()


def show_random_samples(amount=30):
    """
    Attempt to generate images by generating and reconstructing standard normal distributions.
    :param: amount: Amount of samples.
    """
    random_image_samples = amount
    z_generated = np.random.normal(size=(random_image_samples, constants.Z_DIM))
    reconstructed = decoder.predict(np.array(z_generated))

    # Display randomly generated samples.
    fig = plt.figure(figsize=(18, 5), num="Random samples")
    fig.suptitle("Random samples")
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(random_image_samples):
        ax = fig.add_subplot(3, 10, i + 1)
        ax.imshow(reconstructed[i])
    plt.show()


def get_vector_by_label(column, label, neutral_label, batch_size):
    """
    Find vector of given label encoded in latent space.
    :param column: Column to search for in CSV dataset.
    :param label: Given value of column in CSV dataset.
    :param neutral_label: Label that doesn't include target state.
    :param batch_size: Batch size of dataset used to lookup the images with given label.
    :return: A tuple containing unit vector itself and original label value.
    """
    z_file = constants.VECTOR_FOLDER + column + "_" + label
    if path.isfile(z_file + ".npy"):
        print("Vector exists!")
        return column, label, np.load(z_file + ".npy")

    # batch_size = 500.
    data_flow_labeled = imageLoader.build(att, batch_size, label=column)

    current_sum_positive = np.zeros(shape=constants.Z_DIM, dtype="float32")
    total_vectors_with_attribute = 0
    current_mean_positive = np.zeros(shape=constants.Z_DIM, dtype="float32")

    current_sum_negative = np.zeros(shape=constants.Z_DIM, dtype="float32")
    total_vectors_without_attribute = 0
    current_mean_negative = np.zeros(shape=constants.Z_DIM, dtype="float32")

    current_vector = np.zeros(shape=constants.Z_DIM, dtype="float32")
    current_dist = 0

    # Until at least 10 images with the attribute.
    while total_vectors_with_attribute < 10:
        batch = next(data_flow_labeled)
        # (500, 128, 128, 3).
        im = batch[0]
        # (500) because label is defined - 1 attribute for each batch item.
        attribute = batch[1]

        # (500, z_dim)
        z = encoder.predict(np.array(im))

        # Latent vectors of images having given attribute. Example:
        # [z_1, z_2, z_3] chosen using [0, 1, 0] -> z_2 is found.
        latent_vectors_with_attribute = z[attribute == (int(label) if label.isnumeric() else label)]
        latent_vectors_without_attribute = z[attribute == (int(neutral_label) if label.isnumeric() else neutral_label)]

        if len(latent_vectors_with_attribute) > 0:
            current_sum_positive = current_sum_positive + np.sum(latent_vectors_with_attribute, axis=0)
            total_vectors_with_attribute += len(latent_vectors_with_attribute)
            # Average positive vector.
            new_mean_positive = current_sum_positive / total_vectors_with_attribute
            # Size of movement towards positive.
            movement_positive = np.linalg.norm(new_mean_positive - current_mean_positive)

        if len(latent_vectors_without_attribute) > 0:
            current_sum_negative = current_sum_negative + np.sum(latent_vectors_without_attribute, axis=0)
            total_vectors_without_attribute += len(latent_vectors_without_attribute)
            new_mean_negative = current_sum_negative / total_vectors_without_attribute
            movement_negative = np.linalg.norm(new_mean_negative - current_mean_negative)

        # Towards pos and away from neg.
        current_vector = new_mean_positive - new_mean_negative
        # Vector length.
        new_dist = np.linalg.norm(current_vector)
        dist_change = new_dist - current_dist

        print(f"Total {label} images: {str(total_vectors_with_attribute)}.")
        print(f"POS movement: {str(np.round(movement_positive, 3))}.")
        print(f"NEG movement: {str(np.round(movement_negative, 3))}.")
        print(f"Distance: {str(np.round(new_dist, 3))}.")
        print(f"𝛥 Distance: {str(np.round(dist_change, 3))}.")

        current_mean_positive = np.copy(new_mean_positive)
        current_mean_negative = np.copy(new_mean_negative)
        current_dist = np.copy(new_dist)

        # Movement towards positive and negative is insignificant.
        if np.sum([movement_positive, movement_negative]) < 0.08:
            # Convert to unit vector.
            current_vector = current_vector / current_dist
            print("Found the " + label + " vector.")
            break

    np.save(z_file, current_vector)
    return column, label, current_vector


def add_vector_to_images(label_vector, samples_amount, factor_target=5, factor_steps=6):
    """
    Add factorized label's vector to random images and plot continuous transformation.
    :param label_vector: Vector of the label to be added to image.
    :param samples_amount: The amount of images to apply the vector on.
    :param factor_target: Number at which factorization stops (starts at 0).
    :param factor_steps: How quickly factorization reaches factor_target from 0.
    """
    steps = samples_amount
    factors = np.linspace(0, factor_target, factor_steps)
    column, label, vector = label_vector

    att_specific = att[np.logical_not(att[column].isin([int(label) if label.isnumeric() else label]))]
    att_specific = att_specific.reset_index()
    data_flow_specific = imageLoader.build(att_specific, samples_amount)

    example_batch = next(data_flow_specific)
    example_images = example_batch[0]

    z_points = encoder.predict(example_images)

    fig = plt.figure(figsize=(15, 5), num="Vector addition")
    fig2 = plt.figure(figsize=(15, 5), num="Transformation visualization")

    fig.suptitle(column + "=" + label + " vector addition", fontsize=16)
    fig2.suptitle(column + "=" + label + " transformation visualization", fontsize=16)

    counter = 1
    index_increment = 0

    # Main loop - iterate over total amount of samples in one graph.
    for i in range(steps):
        img_source = example_images[i].squeeze()

        sub = fig.add_subplot(steps, len(factors) + 1, counter)
        sub.text(0.5, -0.15, "Original", c="red", fontsize=10, ha="center", transform=sub.transAxes)
        sub.axis("off")

        sub.imshow(img_source)

        counter += 1
        img_prev = None
        img_zero_factor = None

        for j, factor in enumerate(factors):
            changed_z_point = z_points[i] + vector * factor
            changed_image = decoder.predict(np.array([changed_z_point]))[0]

            img = changed_image.squeeze()
            img_zero_factor = img if j == 0 else img_zero_factor

            if img_prev is not None:
                index = counter - ((i + 1) * 2) + index_increment
                sub2 = fig2.add_subplot(steps, len(factors), index)
                sub2.text(
                    0.5,
                    -0.15,
                    "Factor " + str(round(factors[j - 1], 1)) + " -> " + str(round(factors[j], 1)),
                    c="red",
                    fontsize=10,
                    ha="center",
                    transform=sub2.transAxes
                )
                sub2.axis("off")
                diff = keras.backend.mean(keras.backend.square(img - img_prev), axis=2)
                sub2.imshow(diff)

                # On last iteration create absolute difference image.
                if j == len(factors) - 1:
                    index_increment += 1
                    sub3 = fig2.add_subplot(steps, len(factors), index + 1)
                    sub3.text(0.5, -0.15, "Absolute diff", c="purple", fontsize=10, ha="center", transform=sub3.transAxes)
                    sub3.axis("off")

                    diff_absolute = keras.backend.mean(keras.backend.square(img - img_zero_factor), axis=2)
                    sub3.imshow(diff_absolute)

            sub = fig.add_subplot(steps, len(factors) + 1, counter)
            sub.text(0.5, -0.15, "Factor " + str(round(factor, 1)), c="red", fontsize=10, ha="center", transform=sub.transAxes)
            sub.axis("off")
            sub.imshow(img)

            counter += 1
            img_prev = img

    fig.tight_layout()
    fig2.tight_layout()
    plt.show()


def reconstruct_samples(amount):
    """
    Reconstruct image using encoder-decoder forward pass.
    :param amount: Amount of images.
    """
    fig = plt.figure(figsize=(4, 5), num="Sample reconstruction")
    fig.suptitle("Original vs reconstructed image", fontsize=16)
    original_images = data_flow_unlabeled.next()[0][:amount]

    for i in range(amount):
        encoded_images = encoder.predict(original_images)
        decoded_images = decoder.predict(encoded_images)

        sub = fig.add_subplot(amount, 2, (i * 2) + 1)
        sub.imshow(original_images[i])
        sub.axis("off")

        sub2 = fig.add_subplot(amount, 2, (i * 2) + 2)
        sub2.imshow(decoded_images[i])
        sub2.axis("off")
    plt.show()


def morph(start_image_file, end_image_file):
    """
    Vectorize start and end image and plot continuous transformation of one to another.
    :param start_image_file: Name of the starting image from dataset (transform from).
    :param end_image_file: Name of the end image from dataset (transform into).
    """
    factors = np.arange(0, 1, 0.1)

    # Simple way to find csv rows with given image ids.
    att_specific = att[att["Image Index"].isin([start_image_file, end_image_file])]
    # If we set numeric index for all csv rows, flow_from_dataframe will be able to find images
    # corresponding to given row based on the index.
    att_specific = att_specific.reset_index()
    # Just two images.
    data_flow_specific = imageLoader.build(att_specific, 2)

    example_batch = data_flow_specific.next()
    example_images = example_batch[0]

    z_points = encoder.predict(example_images)
    fig = plt.figure(figsize=(18, 8), num="Sample morphing")
    fig.suptitle("Sample morphing", fontsize=16)

    counter = 1

    img = example_images[0].squeeze()
    sub = fig.add_subplot(1, len(factors) + 2, counter)
    sub.axis("off")
    sub.imshow(img)

    counter += 1

    for factor in factors:
        # For early iterations factor is 0 so only start picture is displayed.
        # For later iterations factor is 1 so only end picture is displayed.
        changed_z_point = z_points[0] * (1 - factor) + z_points[1] * factor
        # Returns one batch of results (that's why 0 index).
        changed_image = decoder.predict(np.array([changed_z_point]))[0]

        img = changed_image.squeeze()
        sub = fig.add_subplot(1, len(factors) + 2, counter)
        sub.imshow(img)

        counter += 1

    img = example_images[1].squeeze()
    sub = fig.add_subplot(1, len(factors) + 2, counter)
    sub.imshow(img)

    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument("--vector_transition", help="Visualize continuous vector transition.", action="store_true")
parser.add_argument("--vector_lookup", help="Find and save vectors as numpy arrays.", action="store_true")
parser.add_argument("--vector_reconstruction", help="Reconstruct vectors into original images.", action="store_true")
parser.add_argument("--random_samples", help="Show samples drawn randomly from latent space.", action="store_true")
parser.add_argument("--column", help="CSV column name to seek label in.")
parser.add_argument("--label", help="Value of the column.")
parser.add_argument("--neutral_label", help="Label that doesn't include target state.", type=str, default="No Finding")
parser.add_argument("--f_target", help="Target value of factorization.", type=int, default=5)
parser.add_argument("--f_steps", help="Total amount of factorization steps between 0 and {factor_target}.", type=int,
                    default=6)
parser.add_argument("--samples", help="Amount of samples to display.", type=int, default=1)

args = parser.parse_args()

if args.vector_transition and args.column and args.label:
    print("Vector transition mode launched.")
    print("Simulating " + args.column + "=" + args.label + " vector transition...")
    found_vec = get_vector_by_label(args.column, args.label, args.neutral_label, constants.ANALYSIS_BATCH_SIZE)
    add_vector_to_images(found_vec, args.samples, args.f_target, args.f_steps)
elif args.vector_lookup and args.column and args.label:
    print("Vector lookup mode launched.")
    print("Seeking " + args.column + "=" + str(args.label))
    get_vector_by_label(args.column, args.label, args.neutral_label, 80)
elif args.vector_reconstruction and args.samples:
    print("Vector reconstruction mode launched.")
    reconstruct_samples(args.samples)
elif args.random_samples and args.samples:
    print("Random sampling mode launched.")
    show_random_samples(args.samples)
else:
    print("Illegal argument combination provided.")
