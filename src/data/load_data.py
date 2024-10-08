import tensorflow as tf
from config import TRAIN_DATA_DIR, IMG_SIZE, BATCH_SIZE


def load_data():
    """
    Загружает и подготавливает тренировочный и валидационный генераторы данных из директории изображений

    :return:
        train_generator: Генератор данных для обучения
        validation_generator: Генератор данных для валидации
    :rtype:
        train_generator: Dataset
        validation_generator: Dataset
    """
    train_generator = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DATA_DIR,
        labels='inferred',
        label_mode='categorical',
        shuffle=True,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=42,
        validation_split=0.2,
        subset="training",
    ).repeat().prefetch(buffer_size=tf.data.AUTOTUNE)

    validation_generator = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DATA_DIR,
        labels='inferred',
        label_mode='categorical',
        shuffle=True,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=42,
        validation_split=0.2,
        subset="validation",
    ).repeat().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_generator, validation_generator
