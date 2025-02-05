import tensorflow as tf


def get_augmentation():
    """
    Создает последовательность аугментаций для изображений, включая геометрические и цветовые преобразования

    :return:
        augmentation: Секвенция аугментаций
    :rtype: Sequential
    """
    return tf.keras.Sequential([
        # Геометрические аугментации
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.3),
        tf.keras.layers.RandomTranslation(0.2, 0.2),
        tf.keras.layers.RandomZoom(0.2, 0.2),

        # Цветовые аугментации
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.3),
    ])
