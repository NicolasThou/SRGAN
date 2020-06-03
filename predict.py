from code.train import *


if __name__ == '__main__':

    data_dir = "data/img_align_celeba/*.*"
    batch_size = 3  # 1
    mode = 'predict'

    # Shape of low-resolution and high-resolution images
    low_resolution_shape = (64, 64, 3)
    high_resolution_shape = (256, 256, 3)

    # create the path to the log directory
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # split the dataset between training and test
    X_train, X_test = split_train_test(data_dir, 0.8)

    if mode == 'predict':
        # Build and compile the discriminator network
        discriminator = build_discriminator()

        # Build the generator network
        generator = build_generator()

        # Load models
        generator.load_weights("weights/generator.h5")
        discriminator.load_weights("weights/discriminator.h5")

        # Get 10 random images
        high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=batch_size,
                                                                      low_resolution_shape=(64, 64),
                                                                      high_resolution_shape=(256, 256))
        # Normalize images
        high_resolution_images = high_resolution_images / 127.5 - 1.
        low_resolution_images = low_resolution_images / 127.5 - 1.

        # Generate high-resolution images from low-resolution images
        generated_images = generator.predict_on_batch(low_resolution_images)  # forward

        # Save images
        for index, img in enumerate(generated_images):
            save_images(low_resolution_images[index], high_resolution_images[index], img,
                        path="results/prediction/gen_{}".format(index))