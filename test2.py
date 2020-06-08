from code.train import *
from tensorflow.python.ops.image_ops_impl import psnr
from tensorflow.python.ops.image_ops_impl import ssim
from tensorflow.python.ops.math_ops import reduce_mean as mean

"""
This test for a group of people and a car, to see what if we change the distribution of the test set, different from
the distribution of the image of the training set
"""


def convert_high_low_resolution_test2(images_batch, high_shape, low_shape):

    low_res_images = []
    high_res_images = []

    for index, img in enumerate(images_batch):

        # Get an ndarray of the current image
        img1 = cv2.imread(img)  # BGR image
        img1 = img1.astype(np.float32)

        # Resize the image
        img1_high_resolution = cv2.resize(src=img1, dsize=high_shape, interpolation=cv2.INTER_CUBIC)
        img1_low_resolution = cv2.resize(src=img1, dsize=low_shape, interpolation=cv2.INTER_CUBIC)


        high_res_images.append(img1_high_resolution)
        low_res_images.append(img1_low_resolution)

    return high_res_images, low_res_images


def sample_images_test2(data_dir, batch_size, high_resolution_shape, low_resolution_shape):
    """
    Randomly take images directly from the celebA dataset. Only useful for quick prediction
    """
    # Make a list of all images inside the data directory
    all_images = glob.glob(data_dir)

    # Choose a random batch of images
    images_batch = np.random.choice(all_images, size=batch_size)

    high_resolution_images, low_resolution_images = convert_high_low_resolution_test2(images_batch,
                                                                                high_resolution_shape,
                                                                                low_resolution_shape)

    # Convert the lists to Numpy NDArrays
    return np.array(high_resolution_images), np.array(low_resolution_images)


def save_images_test2(high_resolution, low_resolution, generated_image, path):
    """
    Save low-resolution, high-resolution(original) and
    generated high-resolution images in a single image
    """
    # initialize a figure
    fig = plt.figure()

    # add each image

    ax = fig.add_subplot(1, 3, 1)
    destRGB = cv2.cvtColor(low_resolution, cv2.COLOR_BGR2RGB)
    ax.imshow(destRGB)  # need RGB
    ax.axis("off")
    ax.set_title("Low Resolution")

    ax = fig.add_subplot(1, 3, 2)
    destRGB = cv2.cvtColor(generated_image, cv2.COLOR_BGR2RGB)
    ax.imshow(destRGB)  # need RGB
    ax.axis("off")
    ax.set_title("Generated")

    ax = fig.add_subplot(1, 3, 3)
    destRGB = cv2.cvtColor(high_resolution, cv2.COLOR_BGR2RGB)
    ax.imshow(destRGB)  # need RGB
    ax.axis("off")
    ax.set_title("Original")

    fig.savefig(path)
    plt.close(fig)

if __name__ == '__main__':

    data_dir = "./test2/*.*"
    # X_test set size 40520
    mode = 'test'
    batch_size = 4

    # Shape of low-resolution and high-resolution images
    low_resolution_shape = (64, 64, 3)
    high_resolution_shape = (256, 256, 3)

    if mode == 'test':

        # Build the generator network
        generator = build_generator()

        # Load models
        weight_gen = np.load("weights/generator_6000.npy", allow_pickle=True)
        generator.set_weights(weight_gen)

        # Get random images in the test set
        high_resolution_images, low_resolution_images = sample_images_test2(data_dir=data_dir, batch_size=batch_size,
                                                                                  low_resolution_shape=(64, 64),
                                                                                  high_resolution_shape=(256, 256))
        # Normalize images
        high_resolution_images = high_resolution_images / 127.5 - 1.
        low_resolution_images = low_resolution_images / 127.5 - 1.

        # Generate high-resolution images from low-resolution images
        generated_images = generator.predict_on_batch(low_resolution_images)  # prediction


        # Save images
        for index, img in enumerate(generated_images):
            save_images_test2(high_resolution_images[index], low_resolution_images[index], img,
                        path="test2/results/gen_{}".format(index))