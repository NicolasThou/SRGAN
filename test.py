from code.train import *
from tensorflow.python.ops.image_ops_impl import psnr
from tensorflow.python.ops.image_ops_impl import ssim
from tensorflow.python.ops.math_ops import reduce_mean as mean


def convert_high_low_resolution_test(images_batch, high_shape, low_shape):

    low_res_images = []
    high_res_images = []
    neirest, bicubic, bilinear = [], [], []

    for index, img in enumerate(images_batch):

        # Get an ndarray of the current image
        img1 = cv2.imread(img)  # BGR image
        img1 = img1.astype(np.float32)

        # Resize the image
        img1_high_resolution = cv2.resize(src=img1, dsize=high_shape)
        img1_low_resolution = cv2.resize(src=img1, dsize=low_shape)

        # Save the low resolution
        cv2.imwrite("results/test/Low_res/l_{}.jpg".format(index), img1_low_resolution)

        # from the low, upscale with bicubic, neirest and bilinear
        j = cv2.imread("results/test/Low_res/l_{}.jpg".format(index))  # BGR image
        j = j.astype(np.float32)

        img2_high_resolution = cv2.resize(src=j, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        img3_high_resolution = cv2.resize(src=j, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        img4_high_resolution = cv2.resize(src=j, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)

        high_res_images.append(img1_high_resolution)
        low_res_images.append(img1_low_resolution)
        neirest.append(img2_high_resolution)
        bicubic.append(img3_high_resolution)
        bilinear.append(img4_high_resolution)

    return high_res_images, low_res_images, neirest, bicubic, bilinear


def save_images_test(original_image, generated_image, neirest, bicubic, bilinear, path):
    """
    Save low-resolution, high-resolution(original) and
    generated high-resolution images in a single image
    """
    # initialize a figure
    fig = plt.figure()

    # add each image

    ax = fig.add_subplot(2, 3, 1)
    destRGB = cv2.cvtColor(neirest, cv2.COLOR_BGR2RGB)
    ax.imshow(destRGB)  # need RGB
    ax.axis("off")
    ax.set_title("Nearest Neighbour")

    ax = fig.add_subplot(2, 3, 2)
    destRGB = cv2.cvtColor(bicubic, cv2.COLOR_BGR2RGB)
    ax.imshow(destRGB)  # need RGB
    ax.axis("off")
    ax.set_title("Bicubic")

    ax = fig.add_subplot(2, 3, 3)
    destRGB = cv2.cvtColor(bilinear, cv2.COLOR_BGR2RGB)
    ax.imshow(destRGB)  # need RGB
    ax.axis("off")
    ax.set_title("Bilinear Interpolation")

    ax = fig.add_subplot(2, 3, 4)
    destRGB3 = cv2.cvtColor(generated_image, cv2.COLOR_BGR2RGB)
    ax.imshow(destRGB3)  # need RGB
    ax.axis("off")
    ax.set_title("Generated")

    ax = fig.add_subplot(2, 3, 5)
    destRGB2 = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    ax.imshow(destRGB2)  # need RGB
    ax.axis("off")
    ax.set_title("Original")

    plt.show()
    fig.savefig(path)
    plt.close(fig)


def sample_images_set_test(set, batch_size, high_resolution_shape, low_resolution_shape):
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
    high_resolution_images, low_resolution_images, neirest, bicubic, bilinear = convert_high_low_resolution_test(images_batch,
                                                                                high_resolution_shape,
                                                                                low_resolution_shape)


    # Convert the lists to Numpy NDArrays
    return np.array(high_resolution_images), np.array(low_resolution_images), np.array(neirest), np.array(bicubic), np.array(bilinear)


if __name__ == '__main__':

    mode = 'test'
    batch_size = 1

    # Shape of low-resolution and high-resolution images
    low_resolution_shape = (64, 64, 3)
    high_resolution_shape = (256, 256, 3)

    if mode == 'test':

        # Build the generator network
        generator = build_generator()

        # Load models
        weight_gen = np.load("weights/generator_6000.npy", allow_pickle=True)
        generator.set_weights(weight_gen)


        """
        ================================== 100 images are tested ===========================
        """

        gen_ssim, n_ssim, bicu_ssim, bilin_ssim = [], [], [], []
        gen_psnr, n_psnr, bicu_psnr, bilin_psnr = [], [], [], []

        for i in range(10):

            # Get random images in the test set
            high_resolution_images, low_resolution_images, neirest, bicubic, bilinear = sample_images_set_test(X_test,
                                                                            batch_size=batch_size,
                                                                            low_resolution_shape=(64, 64),
                                                                            high_resolution_shape=(256, 256))

            # Normalize images
            high_resolution_images = high_resolution_images / 127.5 - 1.
            low_resolution_images = low_resolution_images / 127.5 - 1.
            neirest = neirest / 127.5 - 1.
            bicubic = bicubic / 127.5 - 1.
            bilinear = bilinear / 127.5 - 1.


            # Generate high-resolution images from low-resolution images from SRGAN, bicubic interpolation, neirest neighbor
            # bilinear interpolation

            generated_images = generator.predict_on_batch(low_resolution_images)  # prediction

            """
            ============= Compute the average of the 2 metrics PSNR and SSIM over a batch of 10 images ===============
            """

            # convert to tensor
            high = tf.image.convert_image_dtype(high_resolution_images, tf.float32)
            gen = tf.image.convert_image_dtype(generated_images, tf.float32)
            nei = tf.image.convert_image_dtype(neirest, tf.float32)
            bic = tf.image.convert_image_dtype(bicubic, tf.float32)
            bil = tf.image.convert_image_dtype(bilinear, tf.float32)

            # compute the metric for the generator
            first_metric = ssim(high, gen, max_val=1.0)
            gen_ssim.append(first_metric)
            # The last three dimensions of input are expected to be [height, width, depth]
            second_metric = psnr(high, gen, max_val=1.0)
            gen_psnr.append(second_metric)
            print('first metric {}' .format(mean(first_metric)))
            print('second metric {}'.format(mean(second_metric)))

            # compute the metric for the nearest neighbour
            first_metric = ssim(high, nei, max_val=1.0)
            n_ssim.append(first_metric)
            # The last three dimensions of input are expected to be [height, width, depth]
            second_metric = psnr(high, nei, max_val=1.0)
            n_psnr.append(second_metric)
            print('first metric {}'.format(mean(first_metric)))
            print('second metric {}'.format(mean(second_metric)))

            # compute the metric for the bicubic
            first_metric = ssim(high, bic, max_val=1.0)
            bicu_ssim.append(first_metric)
            # The last three dimensions of input are expected to be [height, width, depth]
            second_metric = psnr(high, bic, max_val=1.0)
            bicu_psnr.append(second_metric)
            print('first metric {}'.format(mean(first_metric)))
            print('second metric {}'.format(mean(second_metric)))

            # compute the metric for the bilinear interpolation
            first_metric = ssim(high, bil, max_val=1.0)
            bilin_ssim.append(first_metric)
            # The last three dimensions of input are expected to be [height, width, depth]
            second_metric = psnr(high, bil, max_val=1.0)
            bilin_psnr.append(second_metric)
            print('first metric {}'.format(mean(first_metric)))
            print('second metric {}'.format(mean(second_metric)))

            # Save images
            for index, img in enumerate(generated_images):
                save_images_test(high_resolution_images[index], img, neirest[index],
                                 bicubic[index], bilinear[index], path="results/test/gen_{}_{}".format(i, index))


        print("gen_ssim average : ", np.mean(gen_ssim))
        print("gen_psnr average : ", np.mean(gen_psnr))
        print("n_ssim average : ", np.mean(n_ssim))
        print("n_psnr average : ", np.mean(n_psnr))
        print("bicu_ssim average : ", np.mean(bicu_ssim))
        print("bicu_psnr average : ", np.mean(bicu_psnr))
        print("bilin_ssim average : ", np.mean(bilin_ssim))
        print("bilin_psnr average : ", np.mean(bilin_psnr))