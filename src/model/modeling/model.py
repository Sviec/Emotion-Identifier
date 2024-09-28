import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import EfficientNetV2B2
from src.model.preprocessing.preprocess import get_augmentation


def create_base_model(image_height, image_width):
    base_model = EfficientNetV2B2(
        weights='imagenet',
        include_top=False,
        input_shape=(image_height, image_width, 3),
    )
    return base_model


def build_model(image_height, image_width, num_classes, dropout=0.5):
    base_model = create_base_model(image_height, image_width)

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(image_height, image_width, 3))
    x = get_augmentation()(inputs)

    x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)

    return model
