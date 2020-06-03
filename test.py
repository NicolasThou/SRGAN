from code.train import *
from tensorflow.python.ops.image_ops_impl import psnr
from tensorflow.python.ops.image_ops_impl import ssim
from tensorflow.python.ops.math_ops import reduce_mean as mean

if __name__ == '__main__':

    data_dir = "data/img_align_celeba/*.*"
    # X_test set size 40520
    mode = 'test'
    batch_size = 100

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

        # convert to tensor
        high = tf.image.convert_image_dtype(high_resolution_images, tf.float32)
        gen = tf.image.convert_image_dtype(generated_images, tf.float32)

        # compute the metric for the generator
        first_metric = ssim(high, gen, max_val=1.0)
        # The last three dimensions of input are expected to be [height, width, depth]
        second_metric = psnr(high, gen, max_val=1.0)
        print('first metric {}' .format(mean(first_metric)))
        print('second metric {}'.format(mean(second_metric)))

        # Save images
        for index, img in enumerate(generated_images):
            save_images(low_resolution_images[index], high_resolution_images[index], img,
                        path="results/test/gen_{}".format(index))