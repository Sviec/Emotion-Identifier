from src.model import train, train_fine
from src.scripts import test
from config import EFFICIENTNET

if __name__ == '__main__':
    """
    Запуск всего алгоритма по шагово
    1. Загрузка данных, предобработка данных, создание базовой модели и обучение на замороженных слоях базовой модели
    2. Запуск обучения модели с разморозкой ряда слоев базовой модели
    3. Запуск на тестовом датасете и подготовка файла для отправки на Kaggle
    """
    model, history = train(EFFICIENTNET)
    train_fine(EFFICIENTNET)
    test(EFFICIENTNET)
