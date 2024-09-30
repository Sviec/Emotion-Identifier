from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.data.load_data import load_data
from src.model.modeling.model import build_model
from config import TRAIN_SIZE, BATCH_SIZE, VALIDATION_SIZE, RESULTS_DIR, NUM_CLASSES, EPOCHS, INITIAL_EPOCHS, IMG_SIZE


def freeze_layers(model):
    for layer in model.layers[:198]:
        layer.trainable = False

    for layer in model.layers[346:]:
        layer.trainable = False

    return model


def train_model():
    train_generator, validation_generator = load_data()

    model = build_model(IMG_SIZE[0], IMG_SIZE[1], NUM_CLASSES)

    return model, train_generator, validation_generator


def params_prepare(model_name, lr):
    lr_schedule = ExponentialDecay(lr, decay_steps=100000, decay_rate=0.96)
    optimizer = AdamW(learning_rate=lr_schedule)
    checkpoint = ModelCheckpoint(
        f'{RESULTS_DIR}/{model_name}/checkpoint/{model_name}_ckpt.keras',
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )

    early_stopping = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=4, restore_best_weights=True
    )
    return lr_schedule, optimizer, checkpoint, early_stopping


def fit_model(model_name, model, train_generator, validation_generator, lr=0.0001):
    lr_schedule, optimizer, checkpoint, early_stopping = params_prepare(model_name, lr)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=TRAIN_SIZE // BATCH_SIZE,
        epochs=INITIAL_EPOCHS,
        validation_data=validation_generator,
        validation_steps=VALIDATION_SIZE // BATCH_SIZE,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    return model, history


def fit_fine_model(model_name, model, history, train_generator, validation_generator, lr=0.00001):
    base_model = model.get_layer('efficientnetv2-b2')
    base_model.trainable = True
    freeze_layers(base_model)

    lr_schedule, optimizer, checkpoint, early_stopping = params_prepare(model_name, lr)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    history_fine = model.fit(
        train_generator,
        steps_per_epoch=TRAIN_SIZE // BATCH_SIZE,
        epochs=EPOCHS,
        initial_epoch=len(history.epoch),
        validation_data=validation_generator,
        validation_steps=VALIDATION_SIZE // BATCH_SIZE,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    return model, history_fine
