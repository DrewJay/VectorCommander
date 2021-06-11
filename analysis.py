import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import norm
import settings.constants as constants
import pandas as pd
import argparse
from models.VAE import VariationalAutoencoder
from utils.loaders import ImageLabelLoader

att = pd.read_csv(os.path.join(constants.DATA_FOLDER_NAME, constants.CSV_NAME))

# Exclude channels dim.
imageLoader = ImageLabelLoader(constants.IMAGE_FOLDER, constants.INPUT_DIM[:2])
vae = VariationalAutoencoder.load(VariationalAutoencoder, constants.RUN_FOLDER_NAME)
data_flow_unlabeled = imageLoader.build(att, 10)


def show_distributions():
    """
    Show plot which compares random images encoded into latent vectors
    to standard normal distribution.
    """
    # 20 batches of size 10 = 200 images.
    z_test = vae.encoder.predict_generator(data_flow_unlabeled, steps=20, verbose=1)
    x = np.linspace(-3, 3, 100)
    fig = plt.figure(figsize=(20, 20))
    fig.subplots_adjust(hspace=0.6, wspace=0.4)

    for i in range(50):
        # 5 x 10 grid.
        vector_display = fig.add_subplot(5, 10, i + 1)
        vector_display.hist(z_test[i], density=True, bins=20)
        vector_display.text(0.5, -0.35, "Vector " + str(i), c="red", fontsize=10, ha="center", transform=vector_display.transAxes)
        # pdf returns ys that represent bell curve itself in order, random_normal returns ys that follow
        # particular distribution without order.
        vector_display.plot(x, norm.pdf(x))
    plt.show()


def show_random_samples():
    """
    Attempt to generate images by generating and reconstructing standard normal distributions.
    """
    random_image_samples = 30
    z_generated = np.random.normal(size=(random_image_samples, vae.z_dim))
    reconstructed = vae.decoder.predict(np.array(z_generated))

    # Display randomly generated faces.
    fig = plt.figure(figsize=(18, 5))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(random_image_samples):
        ax = fig.add_subplot(3, 10, i + 1)
        ax.imshow(reconstructed[i])
    plt.show()


def get_vector_by_label(column, label, batch_size):
    """
    Find vector of given label encoded in latent space.
    :param column: Column to search for in CSV dataset.
    :param label: Given value of column in CSV dataset.
    :param batch_size: Batch size of dataset used to lookup the images with given label.
    :return: A tuple containing unit vector itself and original label value.
    """
    # batch_size = 500.
    data_flow_label = imageLoader.build(att, batch_size, label=column)

    current_sum_positive = np.zeros(shape=vae.z_dim, dtype="float32")
    total_vectors_with_attribute = 0
    current_mean_positive = np.zeros(shape=vae.z_dim, dtype="float32")

    current_sum_negative = np.zeros(shape=vae.z_dim, dtype="float32")
    total_vectors_without_attribute = 0
    current_mean_negative = np.zeros(shape=vae.z_dim, dtype="float32")

    current_vector = np.zeros(shape=vae.z_dim, dtype="float32")
    current_dist = 0

    # Until at least 10 images with the attribute.
    while total_vectors_with_attribute < 10:
        batch = next(data_flow_label)
        # (500, 128, 128, 3).
        im = batch[0]
        # (500) because label is defined - 1 attribute for each batch item.
        attribute = batch[1]

        # (500, z_dim)
        z = vae.encoder.predict(np.array(im))
        
        # Latent vectors of images having given attribute. Example:
        # [z_1, z_2, z_3] chosen using [0, 1, 0] -> z_2 is found.
        latent_vectors_with_attribute = z[attribute == label]
        latent_vectors_without_attribute = z[attribute == "No Finding"]

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

    return column, label, current_vector


def add_vector_to_images(label_vector, samples_amount):
    """
    Add factorized label's vector to random images and plot continuous transformation.
    :param label_vector: Vector of the label to be added to image.
    :param samples_amount: The amount of images to apply the vector on.
    """
    steps = samples_amount
    factors = [0, 1, 2, 3, 4, 5]
    column, label, vector = label_vector

    att_specific = att[np.logical_not(att[column].isin([label]))]
    att_specific = att_specific.reset_index()
    data_flow_specific = imageLoader.build(att_specific, samples_amount)

    example_batch = next(data_flow_specific)
    example_images = example_batch[0]

    z_points = vae.encoder.predict(example_images)

    fig = plt.figure(figsize=(18, 10))
    title = fig.add_subplot()
    title.text(0, 1, label + " vector addition", c="black", fontsize=15, transform=title.transAxes)
    title.axis("off")

    counter = 1
    for i in range(steps):
        img = example_images[i].squeeze()

        sub = fig.add_subplot(steps, len(factors) + 1, counter)
        sub.text(0.5, -0.35, "Original", c="red", fontsize=10, ha="center", transform=sub.transAxes)
        sub.axis("off")

        sub.imshow(img)

        counter += 1

        for factor in factors:
            changed_z_point = z_points[i] + vector * factor
            changed_image = vae.decoder.predict(np.array([changed_z_point]))[0]

            img = changed_image.squeeze()
            sub = fig.add_subplot(steps, len(factors) + 1, counter)
            sub.text(0.5, -0.35, "Factor " + str(factor), c="red", fontsize=10, ha="center", transform=sub.transAxes)
            sub.axis("off")
            sub.imshow(img)

            counter += 1

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

    example_batch = next(data_flow_specific)
    example_images = example_batch[0]

    z_points = vae.encoder.predict(example_images)
    figure = plt.figure(figsize=(18, 8))

    counter = 1

    img = example_images[0].squeeze()
    sub = figure.add_subplot(1, len(factors) + 2, counter)
    sub.axis("off")
    sub.imshow(img)

    counter += 1

    for factor in factors:
        # For early iterations factor is 0 so only start picture is displayed.
        # For later iterations factor is 1 so only end picture is displayed.
        changed_z_point = z_points[0] * (1 - factor) + z_points[1] * factor
        # Returns one batch of results (that's why 0 index).
        changed_image = vae.decoder.predict(np.array([changed_z_point]))[0]

        img = changed_image.squeeze()
        sub = figure.add_subplot(1, len(factors) + 2, counter)
        sub.imshow(img)

        counter += 1

    img = example_images[1].squeeze()
    sub = figure.add_subplot(1, len(factors) + 2, counter)
    sub.imshow(img)

    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument("--vector_transition", help="Visualize continuous vector transition")
parser.add_argument("--col", help="CSV column name to seek label in.")
parser.add_argument("--val", help="Value of the column")
parser.add_argument("--samples", help="Choose amount of samples to display", default=3)

args = parser.parse_args()

if args.vector_transition and args.col and args.val:
    print('Vector transition mode launched.')
    print('Simulating ' + args.val + ' vector transition...')
    found_vec = get_vector_by_label(args.col, args.val, constants.ANALYSIS_BATCH_SIZE)
    add_vector_to_images(found_vec, args.samples)
else:
    print('Illegal argument combination provided.')

# py analysis.py --vector_transition 1 --col "Finding Labels" --val "Hernia"
