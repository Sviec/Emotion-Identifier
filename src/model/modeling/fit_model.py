from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.data.load_data import load_data
from src.model.modeling.model import build_model
from config import TRAIN_SIZE, BATCH_SIZE, VALIDATION_SIZE, RESULTS_DIR, NUM_CLASSES, EPOCHS, INITIAL_EPOCHS, IMG_SIZE


def freeze_layers(model):
    """
    Замораживает определенные слои модели для исключения их из процесса обучения

    :param model: Model
        Модель, в которой будут заморожены слои

    :return:
        model: Модель с замороженными слоями
    :rtype: Model

    """
    for layer in model.layers[:198]:
        layer.trainable = False

    for layer in model.layers[346:]:
        layer.trainable = False

    return model


def train_model():
    """
    Подготавливает генераторы данных и создает модель для начального этапа обучения

    :return:
        model: Инициализированная модель для обучения
        train_generator: Генератор обучающих данных
        validation_generator: Генератор данных для валидации
    :rtype:
        model: Model
        train_generator: Dataset
        validation_generator: Dataset

    """

    train_generator, validation_generator = load_data()

    model = build_model(IMG_SIZE[0], IMG_SIZE[1], NUM_CLASSES)

    return model, train_generator, validation_generator


def params_prepare(model_name, lr):
    """
    Подготавливает параметры для обучения модели, включая график изменения скорости обучения, оптимизатор,
    контрольные точки и раннюю остановку

    :param model_name: str
        Название модели, используемое для сохранения контрольных точек
    :param lr: float
        Начальная скорость обучения, которая будет применена в экспоненциальном графике её изменения

    :return:
        lr_schedule: Экспоненциальное снижение скорости обучения
        optimizer: Оптимизатор AdamW с использованием графика скорости обучения
        checkpoint: Объект ModelCheckpoint для сохранения контрольных точек на основе минимальных потерь валидации
        early_stopping: Объект EarlyStopping для остановки обучения, если потери на валидации не улучшаются

    :rtype:
        lr_schedule: ExponentialDecay
            График экспоненциального изменения скорости обучения
        optimizer: AdamW
            Оптимизатор AdamW с графиком скорости обучения
        checkpoint: ModelCheckpoint
            Контрольная точка для сохранения лучшей версии модели
        early_stopping: EarlyStopping
            Механизм ранней остановки обучения, если потери на валидации не улучшаются в течение нескольких эпох

    """
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
    """
     Компилирует и обучает модель на начальном этапе с использованием генераторов данных

    :param model_name: str
        Название модели для сохранения контрольных точек
    :param model: Model
        Модель, которую нужно обучить
    :param train_generator: Dataset
        Генератор обучающих данных
    :param validation_generator: Dataset
        Генератор данных для валидации
    :param lr: float, optional
        Начальная скорость обучения, по умолчанию 0.0001

    :return:
        model: Обученная модель
        history: История обучения модели
    :rtype:
        model: Model
        history: History
    """
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
    """
    Выполняет тонкую настройку модели с использованием заранее обученной базы, замораживая определенные слои

    :param model_name: str
        Название модели для сохранения контрольных точек
    :param model: Model
        Модель, которую нужно дообучить
    :param history: History
        История начального обучения модели
    :param train_generator: Dataset
        Генератор обучающих данных
    :param validation_generator: Dataset
        Генератор данных для валидации
    :param lr: float, optional
        Скорость обучения для тонкой настройки, по умолчанию 0.00001

    :return:
        model: Модель, прошедшая тонкую настройку
        history_fine: История обучения после тонкой настройки
    :rtype:
        model: Model
        history_fine: History
    """
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
