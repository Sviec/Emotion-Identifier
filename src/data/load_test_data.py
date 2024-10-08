import tensorflow as tf
from config import IMG_SIZE, BATCH_SIZE, TEST_DATA_DIR


def load_test_data():
    """
    Загружает тестовые данные из директории для предсказаний без меток.

    :return:
        test_generator: Генератор данных для тестирования (без меток).
    :rtype: Dataset
    """
    return tf.keras.preprocessing.image_dataset_from_directory(
        TEST_DATA_DIR,
        labels=None,
        shuffle=False,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=42,
    )
