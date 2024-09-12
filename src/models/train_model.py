import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.data.load_data import load_data
from src.models.model import build_model, freeze_layers_cns, freeze_layers_en

BATCH_SIZE = 64
TRAIN_SIZE = 40038
VALIDATION_SIZE = 10009


def train_model(model_name):
    image_height, image_width = 224, 224
    initial_epochs = 5
    total_epochs = 30
    num_classes = 9

    train_generator, validation_generator, _ = load_data(BATCH_SIZE, image_height, image_width)

    model = build_model(model_name, image_height, image_width, num_classes)

    return model, train_generator, validation_generator, total_epochs, initial_epochs


def fit_model(model_name, model, train_generator, validation_generator, epochs, lr=0.001, is_freeze=False):
    if is_freeze:
        model.trainable = True
        if model_name == "efficientnet":
            freeze_layers_en(model)
        elif model_name == "convnextsmall":
            freeze_layers_cns(model)

    lr_schedule = ExponentialDecay(lr, decay_steps=100000, decay_rate=0.96)
    optimizer = Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )

    checkpoint = ModelCheckpoint(
        f'../results/{model_name}/checkpoint/{model_name}_ckpt.keras',
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )

    early_stopping = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=4, restore_best_weights=True
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=TRAIN_SIZE // BATCH_SIZE,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=VALIDATION_SIZE // BATCH_SIZE,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    return model, history
