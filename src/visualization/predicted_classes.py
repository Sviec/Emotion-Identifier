import matplotlib.pyplot as plt
from config import VISUAL


def predicted_classes(model, test_generator, class_names):
    """
    Отображает предсказанные классы для выборки изображений из тестового генератора

    :param model: Model
        Обученная модель, используемая для предсказания классов
    :param test_generator: Dataset
        Генератор тестовых данных, возвращающий батчи изображений для предсказаний
    :param class_names: list
        Список имен классов, соответствующих индексам предсказаний

    """
    plt.figure(figsize=(10, 10))
    for batch in test_generator.take(1):
        preds = model.predict(batch).argmax(axis=1)
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(batch[i].numpy().astype("uint8"))
            plt.title(class_names[preds[i]])
            plt.axis("off")

    plt.savefig(f'{VISUAL}/predictions_grid.png')
