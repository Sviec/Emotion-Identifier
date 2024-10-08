from src.model.modeling.model import build_model
from src.data.load_data import load_data
from config import INITIAL_EPOCHS, IMG_SIZE, NUM_CLASSES
from tensorflow.keras.optimizers import AdamW
from hyperopt import fmin, tpe, hp, Trials
import tensorflow as tf
import numpy as np


def optimize_model(params):
    """
    Оптимизирует модель с использованием переданных гиперпараметров и
    возвращает отрицательное значение валидационной точности

    :param params: dict
        Словарь с гиперпараметрами, включая 'learning_rate', 'dropout_rate' и 'batch_size'

    :return:
        val_accuracy: Отрицательное значение максимальной валидационной точности (для минимизации)
    :rtype: float
    """
    train_generator, validation_generator = load_data()

    model = build_model(
        image_height=IMG_SIZE[0],
        image_width=IMG_SIZE[1],
        num_classes=NUM_CLASSES,
        dropout=params['dropout_rate'],
    )

    model.compile(
        optimizer=AdamW(learning_rate=params['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=INITIAL_EPOCHS,
        batch_size=params['batch_size'],
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)]
    )

    val_accuracy = np.max(history.history['val_accuracy'])
    return -val_accuracy


def run_optimization():
    """
    Запускает процесс оптимизации гиперпараметров с использованием метода TPE

    :return:
        best: Лучшие найденные гиперпараметры
    :rtype: dict
    """
    search_space = {
        'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-3)),
        'dropout_rate': hp.uniform('dropout_rate', 0.2, 0.6),
        'batch_size': hp.choice('batch_size', [32, 64]),
    }

    trials = Trials()
    best = fmin(
        fn=optimize_model,
        space=search_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials
    )

    print("Лучшие параметры:", best)

