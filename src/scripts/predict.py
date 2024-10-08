import numpy as np

LABELS_AND_INDICES = (
    ('anger', 0),
    ('contempt', 1),
    ('disgust', 2),
    ('fear', 3),
    ('happy', 4),
    ('neutral', 5),
    ('sad', 6),
    ('surprise', 7),
    ('uncertain', 8),
)

LABELS_TO_INDICES = dict(LABELS_AND_INDICES)
INDICES_TO_LABELS = dict((y, x) for x, y in LABELS_AND_INDICES)


def predictions_to_indices(predictions: np.array):
    """
    Преобразует массив предсказаний вероятностей в массив индексов классов

    :param predictions: np.array
        Массив предсказанных вероятностей для каждого класса

    :return: np.array
        Массив индексов классов, соответствующих максимальным вероятностям
    """
    return predictions.argmax(axis=1)


def indices_to_labels(predictions: np.array):
    """
    Преобразует массив индексов классов в массив меток классов

    :param predictions: np.array
        Массив индексов классов

    :return: np.array
        Массив меток классов, соответствующих индексам
    """
    to_indices = np.vectorize(lambda i: INDICES_TO_LABELS[i])
    return to_indices(predictions)


def predictions_to_labels(predictions: np.array):
    """
    Преобразует массив предсказаний вероятностей в массив меток классов

    :param predictions: np.array
        Массив предсказанных вероятностей для каждого класса

    :return: np.array
        Массив меток классов, соответствующих максимальным вероятностям
    """
    indices = predictions_to_indices(predictions)
    labels = indices_to_labels(indices)
    return labels


def get_predictions(model_name, model, test_generator):
    """
    Получает предсказания модели для тестового генератора и сохраняет их в файл

    :param model_name: str
        Название модели, используемое для сохранения предсказаний
    :param model: Model
        Обученная модель для получения предсказаний
    :param test_generator: Dataset
        Генератор тестовых данных

    :return: np.array
        Массив меток классов, предсказанных моделью
    """
    probs = model.predict(test_generator, verbose=1)
    np.save(f'../results/{model_name}/predictions/predictions.npy', probs)
    probs = np.load(f'../results/{model_name}/predictions/predictions.npy')
    preds = predictions_to_labels(probs)

    return preds


