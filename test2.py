from code.train import *
from tensorflow.python.ops.image_ops_impl import psnr
from tensorflow.python.ops.image_ops_impl import ssim

if __name__ == '__main__':

    data_dir = "data/img_align_celeba/*.*"
    # X_test set size 40520
    mode = 'test'
    batch_size = 3

    # Shape of low-resolution and high-resolution images
    low_resolution_shape = (64, 64, 3)
    high_resolution_shape = (256, 256, 3)

    if mode == 'test':
        # Build and compile the discriminator network
        discriminator = build_discriminator()

        # Build the generator network
        generator = build_generator()

        # Load models
        generator.load_weights("weights/generator.h5")
        discriminator.load_weights("weights/discriminator.h5")

        # Get random images in the test set
        high_resolution_images, low_resolution_images = sample_images_set(X_test, batch_size=batch_size,
                                                                          low_resolution_shape=(64, 64),
                                                                          high_resolution_shape=(256, 256))
        # Normalize images
        high_resolution_images = high_resolution_images / 127.5 - 1.
        low_resolution_images = low_resolution_images / 127.5 - 1.

        # Generate high-resolution images from low-resolution images
        generated_images = generator.predict_on_batch(low_resolution_images)  # prediction

        # ===================

        # transform images to RGB only one image
        destRGB = cv2.cvtColor(low_resolution_images[0], cv2.COLOR_BGR2RGB)
        destRGB2 = cv2.cvtColor(high_resolution_images[0], cv2.COLOR_BGR2RGB)
        destRGB3 = cv2.cvtColor(generated_images[0], cv2.COLOR_BGR2RGB)

        # display images RGB only the first image
        imgplot = plt.imshow(destRGB)
        plt.show()
        imgplot2 = plt.imshow(destRGB2)
        plt.show()
        imgplot3 = plt.imshow(destRGB3)
        plt.show()

        # ===================

        high = tf.convert_to_tensor(high_resolution_images, np.float32)
        gen = tf.convert_to_tensor(generated_images, np.float32)
        # compute the metric
        first_metric = ssim(high, gen, max_val=255)
        second_metric = psnr(high, gen, max_val=255)
        print('first metric {}' .format(first_metric))
        print('second metric {}'.format(second_metric))


        # Save images
        for index, img in enumerate(generated_images):
            save_images(low_resolution_images[index], high_resolution_images[index], img,
                        path="results/test/gen_{}".format(index))