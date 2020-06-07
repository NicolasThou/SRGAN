from code.train import *
from tensorflow.python.ops.image_ops_impl import psnr
from tensorflow.python.ops.image_ops_impl import ssim
from tensorflow.python.ops.math_ops import reduce_mean as mean

if __name__ == '__main__':

    data_dir = "./test2/*.*"
    # X_test set size 40520
    mode = 'test'
    batch_size = 1

    # Shape of low-resolution and high-resolution images
    low_resolution_shape = (64, 64, 3)
    high_resolution_shape = (256, 256, 3)

    if mode == 'test':
        # Build and compile the discriminator network
        discriminator = build_discriminator()

        # Build the generator network
        generator = build_generator()

        # Load models
        weight_gen = np.load("weights/generator_1.npy", allow_pickle=True)
        discri_gen = np.load("weights/discriminator_1.npy", allow_pickle=True)
        generator.set_weights(weight_gen)
        discriminator.set_weights(discri_gen)

        # Get random images in the test set
        high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=3,
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
                        path="test2/results/gen_{}".format(index))