import pickle
from src.model.modeling.fit_model import train_model, fit_model, fit_fine_model
from src.visualization.evaluate import plot_training_history
from src.model.preprocessing.hyperopt import run_optimization
from config import EFFICIENTNET_DIR, EFFICIENTNET_FINE_DIR, HISTORY_DIR
from src.data.load_data import load_data


def train(model_name: str):
    """
    Функция запускает загрузку данных, обучает модель, строит график динамики точности и валидационной ошибки,
    сохраняет в формате keras в файлы проекта и записывает историю обучению

    :param model_name: Наименование модели
    :type model_name: str

    :return: Созданная модель, история обучения
    :rtype: model: Model
    :rtype: history: History
    """
    model, train_generator, validation_generator = train_model()
    model, history = fit_model(model_name, model, train_generator, validation_generator)

    plot_training_history(history)

    model.save(EFFICIENTNET_DIR)

    with open(f'{HISTORY_DIR}/history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    return model, history


def train_fine(model_name: str, model, history):
    """
    Запуск Fine Tuning, сохранение обученной модели в файлы проекта

    :param model_name: str
    :param model: Model
    :param history: History
    """
    train_generator, validation_generator = load_data()

    model_fine, history_fine = fit_fine_model(
        model_name, model, history,
        train_generator, validation_generator
    )
    plot_training_history(history_fine, True)

    model.save(EFFICIENTNET_FINE_DIR)