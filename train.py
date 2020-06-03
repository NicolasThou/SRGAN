import glob
import random
import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Input
from keras.applications import VGG19
from keras.layers import BatchNormalization, Activation, LeakyReLU, Add, Dense
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Model
from tensorflow.keras.optimizers import Adam
import cv2
from torch.utils.tensorboard import SummaryWriter
""" ======== MacOS issue, comment if you don't need it ======= """
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
""" ========================================================== """


def residual_block(x):
    """
    Residual block
    """
    filters = [64, 64]
    kernel_size = 3
    strides = 1
    padding = "same"
    momentum = 0.8
    activation = "relu"

    res = Conv2D(filters=filters[0], kernel_size=kernel_size, strides=strides, padding=padding)(x)
    res = Activation(activation=activation)(res)
    res = BatchNormalization(momentum=momentum)(res)

    res = Conv2D(filters=filters[1], kernel_size=kernel_size, strides=strides, padding=padding)(res)
    res = BatchNormalization(momentum=momentum)(res)

    # Add res and x
    res = Add()([res, x])
    return res


def build_generator():
    """
    Create a generator network using the hyperparameter values defined below
    :return:
    """
    residual_blocks = 16
    momentum = 0.8
    input_shape = (64, 64, 3)

    # Input Layer of the generator network
    input_layer = Input(shape=input_shape)

    # Add the pre-residual block
    gen1 = Conv2D(filters=64, kernel_size=9, strides=1, padding='same', activation='relu')(input_layer)

    # Add 16 residual blocks
    res = residual_block(gen1)
    for i in range(residual_blocks - 1):
        res = residual_block(res)

    # Add the post-residual block
    gen2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(res)
    gen2 = BatchNormalization(momentum=momentum)(gen2)

    # Take the sum of the output from the pre-residual block(gen1) and the post-residual block(gen2)
    gen3 = Add()([gen2, gen1])

    # Add an upsampling block
    gen4 = UpSampling2D(size=2)(gen3)
    gen4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen4)
    gen4 = Activation('relu')(gen4)

    # Add another upsampling block
    gen5 = UpSampling2D(size=2)(gen4)
    gen5 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen5)
    gen5 = Activation('relu')(gen5)

    # Output convolution layer
    gen6 = Conv2D(filters=3, kernel_size=9, strides=1, padding='same')(gen5)
    output = Activation('tanh')(gen6)

    # Keras model
    model = Model(inputs=[input_layer], outputs=[output], name='generator')
    return model


def build_discriminator():
    """
    Create a discriminator network using the hyperparameter values defined below
    :return:
    """
    leakyrelu_alpha = 0.2
    momentum = 0.8
    input_shape = (256, 256, 3)

    input_layer = Input(shape=input_shape)

    # Extractor features

    # Add the first convolution block
    dis1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(input_layer)
    dis1 = LeakyReLU(alpha=leakyrelu_alpha)(dis1)

    # Add the 2nd convolution block
    dis2 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(dis1)
    dis2 = LeakyReLU(alpha=leakyrelu_alpha)(dis2)
    dis2 = BatchNormalization(momentum=momentum)(dis2)

    # Add the third convolution block
    dis3 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(dis2)
    dis3 = LeakyReLU(alpha=leakyrelu_alpha)(dis3)
    dis3 = BatchNormalization(momentum=momentum)(dis3)

    # Add the fourth convolution block
    dis4 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(dis3)
    dis4 = LeakyReLU(alpha=leakyrelu_alpha)(dis4)
    dis4 = BatchNormalization(momentum=0.8)(dis4)

    # Add the fifth convolution block
    dis5 = Conv2D(256, kernel_size=3, strides=1, padding='same')(dis4)
    dis5 = LeakyReLU(alpha=leakyrelu_alpha)(dis5)
    dis5 = BatchNormalization(momentum=momentum)(dis5)

    # Add the sixth convolution block
    dis6 = Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(dis5)
    dis6 = LeakyReLU(alpha=leakyrelu_alpha)(dis6)
    dis6 = BatchNormalization(momentum=momentum)(dis6)

    # Add the seventh convolution block
    dis7 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(dis6)
    dis7 = LeakyReLU(alpha=leakyrelu_alpha)(dis7)
    dis7 = BatchNormalization(momentum=momentum)(dis7)

    # Add the eight convolution block
    dis8 = Conv2D(filters=512, kernel_size=3, strides=2, padding='same')(dis7)
    dis8 = LeakyReLU(alpha=leakyrelu_alpha)(dis8)
    dis8 = BatchNormalization(momentum=momentum)(dis8)

    # fully connected layers to classify the image

    # Add a dense layer
    dis9 = Dense(units=1024)(dis8)
    dis9 = LeakyReLU(alpha=0.2)(dis9)

    # Last dense layer - for classification
    output = Dense(units=1, activation='sigmoid')(dis9)

    model = Model(inputs=[input_layer], outputs=[output], name='discriminator')
    return model


def build_vgg():
    """
    Build VGG network to extract image features
    """
    input_shape = (256, 256, 3)

    # Load a pre-trained VGG19 model trained on 'Imagenet' dataset
    vgg = VGG19(weights="imagenet")
    vgg.outputs = [vgg.layers[9].output]

    input_layer = Input(shape=input_shape)

    # Extract features
    features = vgg(input_layer)

    # Create a Keras model
    model = Model(inputs=[input_layer], outputs=[features])
    return model


def convert_high_low_resolution(images_batch, high_shape, low_shape):
    """
    This function maps for each images of the batch a downsampling in order to
    have high resolution image list and a low resolution image list during the training or the test

    :param images_batch: batch of images
    :param high_shape: shape tuple of size 2
    :param low_shape: shape tuple of size 2

    :return: lists
    """

    low_res_images = []
    high_res_images = []

    for img in images_batch:
        # Get an ndarray of the current image
        img1 = cv2.imread(img)  # BGR image
        img1 = img1.astype(np.float32)

        # Resize the image
        img1_high_resolution = cv2.resize(src=img1, dsize=high_shape)
        img1_low_resolution = cv2.resize(src=img1, dsize=low_shape)

        # Do a random horizontal flip
        if np.random.random() < 0.5:
            img1_high_resolution = np.fliplr(img1_high_resolution)
            img1_low_resolution = np.fliplr(img1_low_resolution)

        high_res_images.append(img1_high_resolution)
        low_res_images.append(img1_low_resolution)

    return high_res_images, low_res_images



def sample_images(data_dir, batch_size, high_resolution_shape, low_resolution_shape):
    """
    Randomly take images directly from the celebA dataset. Only useful for quick prediction
    """
    # Make a list of all images inside the data directory
    all_images = glob.glob(data_dir)

    # Choose a random batch of images
    images_batch = np.random.choice(all_images, size=batch_size)

    high_resolution_images, low_resolution_images = convert_high_low_resolution(images_batch,
                                                                                high_resolution_shape,
                                                                                low_resolution_shape)

    # Convert the lists to Numpy NDArrays
    return np.array(high_resolution_images), np.array(low_resolution_images)


def split_train_test(data_directory, split):
    """
    This function splits the dataset into two separate sets for the training and the test

    :param data_directory: path to the dataset
    :param split: purcent of images for the training set


    :return: list : X_train list of images for the training set
                    X_test list of images for the test set

    """
    if split > 1 or split < 0:
        raise ValueError("split has to be between 0 and 1")

    # Make a list of all images inside the data directory
    all_images = glob.glob(data_directory)
    n = len(all_images)  # 202 599

    # Split train-test list of image
    l_boundary = int(split * n)  # for X_train 162 079 images, and X_test 40 520 images
    X_train_images = all_images[:l_boundary]
    X_test_images = all_images[l_boundary:]

    # Shuffle X_train and X_test
    random.shuffle(X_train_images)
    random.shuffle(X_test_images)

    return X_train_images, X_test_images


def sample_images_set(set, batch_size, high_resolution_shape, low_resolution_shape):
    """
    Sample the batch according to the set (training or test)
    Convert the batch in low and high resolution images

    :param set: training set or test set
    :param batch_size: size of the batch
    :param high_resolution_shape: tuple size 2
    :param low_resolution_shape: tuple size 2

    :return: 2 ndarray of high and low resolution
    """

    # Choose a batch of images randomly
    images_batch = np.random.choice(set, size=batch_size)

    # Convert the images into two array of high and low resolution images from the batch
    high_resolution_images, low_resolution_images = convert_high_low_resolution(images_batch,
                                                                                high_resolution_shape,
                                                                                low_resolution_shape)


    # Convert the lists to Numpy NDArrays
    return np.array(high_resolution_images), np.array(low_resolution_images)


def save_images(low_resolution_image, original_image, generated_image, path):
    """
    Save low-resolution, high-resolution(original) and
    generated high-resolution images in a single image
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    destRGB = cv2.cvtColor(low_resolution_image, cv2.COLOR_BGR2RGB)
    ax.imshow(destRGB)  # need RGB
    ax.axis("off")
    ax.set_title("Low-resolution")

    ax = fig.add_subplot(1, 3, 2)
    destRGB2 = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    ax.imshow(destRGB2)  # need RGB
    ax.axis("off")
    ax.set_title("Original")

    ax = fig.add_subplot(1, 3, 3)
    destRGB3 = cv2.cvtColor(generated_image, cv2.COLOR_BGR2RGB)
    ax.imshow(destRGB3)  # need RGB
    ax.axis("off")
    ax.set_title("Generated")

    plt.savefig(path)

data_dir = "data/img_align_celeba/*.*"
# split the dataset between training and test
X_train, X_test = split_train_test(data_dir, 0.8)

if __name__ == '__main__':

    # X_train training set size 162 080
    # per epoch, all the training set has to be treated
    # number_of_batch * batch_size = size_training_set
    number_of_epochs = 2  # 30 000
    number_of_batch = 2  # 5065, 2532, 1266 according to the batch size
    batch_size = 2  # 32, 64, 128
    mode = 'train'

    # Shape of low-resolution and high-resolution images
    low_resolution_shape = (64, 64, 3)
    high_resolution_shape = (256, 256, 3)

    # Common optimizer for all networks
    common_optimizer = Adam(0.0002, 0.5)

    # Add Tensorboard
    writter = SummaryWriter()

    if mode == 'train':

        # Build and compile VGG19 network to extract features
        vgg = build_vgg()
        # don't train vgg, it's a pre-trained model
        vgg.trainable = False
        vgg.compile(loss='mse', optimizer=common_optimizer, metrics=['accuracy'])

        # Build and compile the discriminator network. The discriminator will be train.
        discriminator = build_discriminator()
        # pixel wise MSE Loss for the discriminator, LSGAN against Vanishing gradient
        # 1/2 * ((D(x) - 1)^2) + 1/2 * (((D(G(z))) - 0) ^ 2)
        discriminator.compile(loss='mse', optimizer=common_optimizer)

        # Build the generator network
        generator = build_generator()

        # Build and compile the adversarial model.

        # Input layers for high-resolution and low-resolution images
        input_high_resolution = Input(shape=high_resolution_shape)
        input_low_resolution = Input(shape=low_resolution_shape)

        # Generate high-resolution images from low-resolution images
        generated_high_resolution_images = generator(input_low_resolution)

        # Extract feature maps of the generated images
        features = vgg(generated_high_resolution_images)

        # Make the discriminator network as non-trainable, we don't want to train the discriminator during
        # the training of the SRGAN
        discriminator.trainable = False

        # Get the probability of generated high-resolution images
        probs = discriminator(generated_high_resolution_images)

        # Create an adversarial model, a Model Keras that take as input : [input_low_resolution, input_high_resolution]
        # and output : probability of the discriminator, feature map from pre-trained VGG
        adversarial_model = Model([input_low_resolution, input_high_resolution], [probs, features])

        # According to the paper, 0.001 * Loss_Discriminator_SRGAN + 1 * Loss_VGG
        # the adversarial model try to minimize the above loss to output realistic image
        adversarial_model.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 1],
                                  optimizer=common_optimizer)

        for epoch in range(number_of_epochs):
            print("Epoch:{}".format(epoch))
            g_loss_epoch, d_loss_epoch = [], []
            for batch_index in range(number_of_batch):
                print("Batch index:{}".format(batch_index))

                """ ================================================================================="""
                """ ============================= TRAIN DISCRIMINATOR ==============================="""
                """ ================================================================================="""

                # Sample a batch of images
                high_resolution_images, low_resolution_images = sample_images_set(X_train, batch_size=batch_size,
                                                                                  low_resolution_shape=(64, 64),
                                                                                  high_resolution_shape=(256, 256),
                                                                                  )

                # Normalize images
                high_resolution_images = high_resolution_images / 127.5 - 1.
                low_resolution_images = low_resolution_images / 127.5 - 1.

                # Generate high-resolution images from low-resolution images
                generated_high_resolution_images = generator.predict(low_resolution_images)

                # Generate batch of real and fake labels, true set to 1, and false set to 0
                real_labels = np.ones((batch_size, 16, 16, 1))
                fake_labels = np.zeros((batch_size, 16, 16, 1))

                # Train the discriminator network on real and fake images
                # list of scalars (if the model has multiple outputs and/or metrics)
                # the discriminator take as input an image, output probabilities, and has to be as close as possible
                # to the target labels
                d_loss_real = discriminator.train_on_batch(high_resolution_images, real_labels)  # forward back propagation
                d_loss_fake = discriminator.train_on_batch(generated_high_resolution_images, fake_labels)  # forward back propagation

                # Calculate total discriminator loss (LSGAN Discriminator)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                d_loss_epoch.append(d_loss)
                print("d_loss:", d_loss)

                """ ================================================================================="""
                """ ============================= TRAIN Generator ==================================="""
                """ ================================================================================="""

                # Sample a batch of images
                high_resolution_images, low_resolution_images = sample_images_set(X_train, batch_size=batch_size,
                                                                                        low_resolution_shape=(64, 64),
                                                                                        high_resolution_shape=(256, 256),
                                                                                        )

                # Normalize images
                high_resolution_images = high_resolution_images / 127.5 - 1.
                low_resolution_images = low_resolution_images / 127.5 - 1.

                # Extract feature maps for real high-resolution images for the target
                image_features = vgg.predict(high_resolution_images)

                # Train the generator network
                # Input-Training data : [low_resolution_images, high_resolution_images]
                # Target data : [real_labels, image_features]
                # The target of the model is to be classify as real image from the discriminator, and to have a generated
                # image that have the same feature map than the high_resolution_images (from VGG)
                g_loss = adversarial_model.train_on_batch([low_resolution_images, high_resolution_images],
                                                 [real_labels, image_features])  # forward back propagation
                g_loss_epoch.append(g_loss[0])

                # [final_adversarial loss, binary_cross_entropy, mse]
                print("g_loss:", g_loss)

                """ ================================================================================="""
                """ ================== save model weight and save image ============================="""
                """ ================================================================================="""


                # Sample and save images after every 100 batches
                if batch_index % 100 == 0:
                    high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=batch_size,
                                                                                  low_resolution_shape=(64, 64),
                                                                                  high_resolution_shape=(256, 256))
                    # Normalize images
                    high_resolution_images = high_resolution_images / 127.5 - 1.
                    low_resolution_images = low_resolution_images / 127.5 - 1.

                    generated_images = generator.predict_on_batch(low_resolution_images)

                    for index, img in enumerate(generated_images):
                        save_images(low_resolution_images[index], high_resolution_images[index], img,
                                    path="results/training/img_{}_{}_{}".format(epoch, number_of_batch, index))

            # Write the losses to Tensorboard

            writter.add_scalar('train_adversarial_loss', np.mean(g_loss_epoch), epoch)
            writter.add_scalar('train_discriminator_loss', np.mean(d_loss_epoch), epoch)

        # Save models
        generator.save_weights("weights/generator.h5")
        discriminator.save_weights("weights/discriminator.h5")



