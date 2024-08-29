import tensorflow as tf


def load_data(batch_size, image_height, image_width, total_epochs, data_dir='data/raw'):
    train_generator = tf.keras.preprocessing.image_dataset_from_directory(
        f'../{data_dir}/train',
        labels='inferred',
        label_mode='categorical',
        shuffle=True,
        image_size=(image_height, image_width),
        batch_size=batch_size,
        seed=42,
        validation_split=0.2,
        subset="training",
    )
    train_generator = train_generator.repeat(total_epochs)
    validation_generator = tf.keras.preprocessing.image_dataset_from_directory(
        f'../{data_dir}/train',
        labels='inferred',
        label_mode='categorical',
        shuffle=True,
        image_size=(image_height, image_width),
        batch_size=batch_size,
        seed=42,
        validation_split=0.2,
        subset="validation",
    )

    test_generator = tf.keras.preprocessing.image_dataset_from_directory(
        f'../{data_dir}/test_kaggle',
        labels=None,
        shuffle=False,
        image_size=(image_height, image_width),
        batch_size=batch_size,
        seed=42,
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_generator = train_generator.prefetch(buffer_size=AUTOTUNE)
    validation_generator = validation_generator.prefetch(buffer_size=AUTOTUNE)

    return train_generator, validation_generator, test_generator
