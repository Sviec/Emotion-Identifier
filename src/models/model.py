import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import EfficientNetV2B2, ConvNeXtSmall
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import RandomFlip, RandomRotation


def create_en(image_height, image_width):
    base_model = EfficientNetV2B2(
        weights='imagenet',
        include_top=False,
        input_shape=(image_height, image_width, 3),
    )
    return base_model


def create_cns(image_height, image_width):
    base_model = ConvNeXtSmall(
        weights='imagenet',
        include_top=False,
        input_shape=(image_height, image_width, 3),
    )
    return base_model


def freeze_layers_cns(model):
    for layer in model.layers[:237]:
        layer.trainable = False

    for layer in model.layers:
        if 'layer_normalization' in layer.name:
            layer.trainable = False

    return model


def freeze_layers_en(model):
    for layer in model.layers[:198]:
        layer.trainable = False

    for layer in model.layers[346:]:
        layer.trainable = False

    return model


def build_model(model_name, image_height, image_width, num_classes):
    if model_name == "efficientnet":
        base_model = create_en(image_height, image_width)
    elif model_name == "convnextsmall":
        base_model = create_cns(image_height, image_width)
    else:
        return

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(image_height, image_width, 3))
    x = RandomFlip('horizontal')(inputs)
    x = RandomRotation(0.2)(x)
    if model_name == "efficientnet":
        x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)

    return model
