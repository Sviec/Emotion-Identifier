
# Параметры обучения
IMG_SIZE = (224, 224)
NUM_CLASSES = 9
INITIAL_EPOCHS = 10
EPOCHS = 35
BATCH_SIZE = 64
TRAIN_SIZE = 40038
VALIDATION_SIZE = 10009

# model
EFFICIENTNET = "efficientnet"

# Пути
RESULTS_DIR = "../results"
TRAIN_DATA_DIR = "../data/raw/train"
TEST_DATA_DIR = "../data/raw/test_kaggle"
EFFICIENTNET_DIR = f"../results/{EFFICIENTNET}/emotion_model.keras"
EFFICIENTNET_FINE_DIR = f"../results/{EFFICIENTNET}/emotion_fine_model.keras"
VISUAL = f"../results/{EFFICIENTNET}/visualize"
SOLUTION = f"../results/{EFFICIENTNET}/solution"
HISTORY_DIR = f"../results/{EFFICIENTNET}/history"


# Гиперпараметры
