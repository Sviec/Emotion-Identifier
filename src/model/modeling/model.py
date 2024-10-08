import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import EfficientNetV2B2
from src.model.preprocessing.preprocess import get_augmentation


def create_base_model(image_height, image_width):
    """
    Создает базовую модель на основе EfficientNetV2B2 с предобученными весами на ImageNet без верхних слоев

    :param image_height: int
      Высота входного изображения
    :param image_width: int
      Ширина входного изображения

    :return:
      base_model: Базовая модель EfficientNetV2B2
    :rtype: Model
    """
    base_model = EfficientNetV2B2(
        weights='imagenet',
        include_top=False,
        input_shape=(image_height, image_width, 3),
    )
    return base_model


def build_model(image_height, image_width, num_classes):
    """
    Создает модель классификации изображений на основе EfficientNetV2

    :param image_height: int
        Высота входных изображений
    :param image_width: int
        Ширина входных изображений
    :param num_classes: int
        Количество классов для классификации

    :return: Model
        Созданная модель с заданной архитектурой
    """
    base_model = create_base_model(image_height, image_width)

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(image_height, image_width, 3))
    x = get_augmentation()(inputs)

    x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)

    return model
