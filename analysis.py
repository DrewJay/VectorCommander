import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import norm
import pandas as pd
from models.VAE import VariationalAutoencoder
from utils.loaders import load_model, ImageLabelLoader


# Create folder run/vae/${run_id}_${data_name}/viz, images, weights
section = "vae"
run_id = "0001"
data_name = "faces"
RUN_FOLDER = "run/%s/" % section
RUN_FOLDER += "_".join([run_id, data_name])  # Array means join from left and right.

DATA_FOLDER = "./data/train/"
IMAGE_FOLDER = "./data/train/images/"

INPUT_DIM = (128, 128, 3)
att = pd.read_csv(os.path.join(DATA_FOLDER, "list_attr_celeba.csv"))

# Exclude channels dim.
imageLoader = ImageLabelLoader(IMAGE_FOLDER, INPUT_DIM[:2])
vae = load_model(VariationalAutoencoder, RUN_FOLDER)


n_to_show = 10
data_flow_generic = imageLoader.build(att, n_to_show)
example_batch = next(data_flow_generic)
example_images = example_batch[0]

z_points = vae.encoder.predict(example_images)
reconst_images = vae.decoder.predict(z_points)

fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(n_to_show):
    img = example_images[i].squeeze()
    sub = fig.add_subplot(2, n_to_show, i + 1)
    sub.axis("off")        
    sub.imshow(img)

for i in range(n_to_show):
    img = reconst_images[i].squeeze()
    sub = fig.add_subplot(2, n_to_show, i + n_to_show + 1)
    sub.axis("off")
    sub.imshow(img)


# 20 batches of size 10 = 200 images.
z_test = vae.encoder.predict_generator(data_flow_generic, steps=20, verbose=1)
x = np.linspace(-3, 3, 100)
fig = plt.figure(figsize=(20, 20))
fig.subplots_adjust(hspace=0.6, wspace=0.4)

for i in range(50):
    ax = fig.add_subplot(5, 10, i + 1)
    # z_test[:,i] = (200)
    # Done like this probably because if [[z_1], [z_2], [z_3]] unrelated latent vectors are supposed
    # to be standard normal distributions, building distribution like [z_1[i], z_2[i], z_3[i]]
    # will verify this property better.
    ax.hist(z_test[:,i], density=True, bins=20)
    ax.axis("off")
    ax.text(0.5, -0.35, str(i), fontsize=10, ha="center", transform=ax.transAxes)
    # pdf returns ys that represent bell curve itself in order, random_normal returns ys that follow
    # particular distribution without order.
    ax.plot(x, norm.pdf(x))

plt.show()


n_to_show = 30
znew = np.random.normal(size=(n_to_show, vae.z_dim))
reconst = vae.decoder.predict(np.array(znew))

fig = plt.figure(figsize=(18, 5))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(n_to_show):
    ax = fig.add_subplot(3, 10, i + 1)
    ax.imshow(reconst[i, :, :, :])
    ax.axis("off")

plt.show()


def get_vector_from_label(label, batch_size):
    # batch_size = 500
    data_flow_label = imageLoader.build(att, batch_size, label=label)

    current_sum_POS = np.zeros(shape=vae.z_dim, dtype="float32")
    current_n_POS = 0
    current_mean_POS = np.zeros(shape=vae.z_dim, dtype="float32")

    current_sum_NEG = np.zeros(shape=vae.z_dim, dtype="float32")
    current_n_NEG = 0
    current_mean_NEG = np.zeros(shape=vae.z_dim, dtype="float32")

    current_vector = np.zeros(shape=vae.z_dim, dtype="float32")
    current_dist = 0

    print("label: " + label)
    print("images : POS move : NEG move :distance : 𝛥 distance")

    # Until atleast 10 images with the attribute.
    while(current_n_POS < 10):
        batch = next(data_flow_label)
        # (500, 128, 128, 3)
        im = batch[0]
        # (500) because label is defined - 1 attribute for each batch item.
        attribute = batch[1]
        print(np.shape(attribute))

        # (500, z_dim)
        z = vae.encoder.predict(np.array(im))
        
        # Latent vectors of images having given attribute. Example:
        # [z_1, z_2, z_3] chosen using [0, 1, 0] -> z_2 is found.
        z_POS = z[attribute == 1]
        z_NEG = z[attribute == -1]

        if len(z_POS) > 0:
            current_sum_POS = current_sum_POS + np.sum(z_POS, axis=0)
            current_n_POS += len(z_POS)
            # Average positive vector.
            new_mean_POS = current_sum_POS / current_n_POS
            # Size of movement towards positive.
            movement_POS = np.linalg.norm(new_mean_POS - current_mean_POS)

        if len(z_NEG) > 0:
            current_sum_NEG = current_sum_NEG + np.sum(z_NEG, axis=0)
            current_n_NEG += len(z_NEG)
            new_mean_NEG = current_sum_NEG / current_n_NEG
            movement_NEG = np.linalg.norm(new_mean_NEG - current_mean_NEG)

        # Towards pos and away from neg.
        current_vector = new_mean_POS - new_mean_NEG
        # Vector length.
        new_dist = np.linalg.norm(current_vector)
        dist_change = new_dist - current_dist

        print(str(current_n_POS)
              + "    : " + str(np.round(movement_POS, 3))
              + "    : " + str(np.round(movement_NEG, 3))
              + "    : " + str(np.round(new_dist, 3))
              + "    : " + str(np.round(dist_change, 3))
             )

        current_mean_POS = np.copy(new_mean_POS)
        current_mean_NEG = np.copy(new_mean_NEG)
        current_dist = np.copy(new_dist)

        # Movement towards positive and negative is insignificant.
        if np.sum([movement_POS, movement_NEG]) < 0.08:
            # Convert to unit vector
            current_vector = current_vector / current_dist
            print("Found the " + label + " vector")
            break

    return current_vector   


def add_vector_to_images(feature_vec):
    n_to_show = 5
    factors = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

    example_batch = next(data_flow_generic)
    example_images = example_batch[0]
    example_labels = example_batch[1]

    z_points = vae.encoder.predict(example_images)

    fig = plt.figure(figsize=(18, 10))

    counter = 1

    for i in range(n_to_show):
        img = example_images[i].squeeze()
        sub = fig.add_subplot(n_to_show, len(factors) + 1, counter)
        sub.axis("off")        
        sub.imshow(img)

        counter += 1

        for factor in factors:
            changed_z_point = z_points[i] + feature_vec * factor
            changed_image = vae.decoder.predict(np.array([changed_z_point]))[0]

            img = changed_image.squeeze()
            sub = fig.add_subplot(n_to_show, len(factors) + 1, counter)
            sub.axis("off")
            sub.imshow(img)

            counter += 1

    plt.show()


BATCH_SIZE = 500
attractive_vec = get_vector_from_label("Attractive", BATCH_SIZE)
mouth_open_vec = get_vector_from_label("Mouth_Slightly_Open", BATCH_SIZE)
smiling_vec = get_vector_from_label("Smiling", BATCH_SIZE)
lipstick_vec = get_vector_from_label("Wearing_Lipstick", BATCH_SIZE)
young_vec = get_vector_from_label("High_Cheekbones", BATCH_SIZE)
male_vec = get_vector_from_label("Male", BATCH_SIZE)
eyeglasses_vec = get_vector_from_label("Eyeglasses", BATCH_SIZE)
blonde_vec = get_vector_from_label("Blond_Hair", BATCH_SIZE)


print("Attractive Vector")
add_vector_to_images(attractive_vec)

print("Mouth Open Vector")
add_vector_to_images(mouth_open_vec)

print("Smiling Vector")
add_vector_to_images(smiling_vec)

print("Lipstick Vector")
add_vector_to_images(lipstick_vec)

print("Young Vector")
add_vector_to_images(young_vec)

print("Male Vector")
add_vector_to_images(male_vec)

print("Eyeglasses Vector")
add_vector_to_images(eyeglasses_vec)

print("Blond Vector")
add_vector_to_images(blonde_vec)


def morph_faces(start_image_file, end_image_file):
    factors = np.arange(0, 1, 0.1)

    # Simple way to find csv rows with given image ids.
    att_specific = att[att["image_id"].isin([start_image_file, end_image_file])]
    # If we set numeric index for all csv rows, flow_from_dataframe will be able to find images
    # corresponding to given row based on the index.
    att_specific = att_specific.reset_index()
    # Just two images.
    data_flow_label = imageLoader.build(att, 2)

    example_batch = next(data_flow_label)
    example_images = example_batch[0]
    example_labels = example_batch[1]

    z_points = vae.encoder.predict(example_images)

    fig = plt.figure(figsize=(18, 8))

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
        changed_image = vae.decoder.predict(np.array([changed_z_point]))[0]

        img = changed_image.squeeze()
        sub = fig.add_subplot(1, len(factors) + 2, counter)
        sub.axis("off")
        sub.imshow(img)

        counter += 1

    img = example_images[1].squeeze()
    sub = fig.add_subplot(1, len(factors) + 2, counter)
    sub.axis("off")        
    sub.imshow(img)

    plt.show()


start_image_file = "000238.jpg"
end_image_file = "000193.jpg"

morph_faces(start_image_file, end_image_file)

start_image_file = "000112.jpg"
end_image_file = "000258.jpg"

morph_faces(start_image_file, end_image_file)

start_image_file = "000230.jpg"
end_image_file = "000712.jpg"

morph_faces(start_image_file, end_image_file)

