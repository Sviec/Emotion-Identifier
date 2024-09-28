import tensorflow as tf
from src.scripts.predict import get_predictions
from src.data.load_test_data import load_test_data
from src.scripts.send_to_kaggle import send_to_kaggle
from config import EFFICIENTNET_DIR


def test(model_name):
    test_generator = load_test_data()
    model = tf.keras.models.load_model(f'{EFFICIENTNET_DIR}')
    preds = get_predictions(model_name, model, test_generator)
    send_to_kaggle(test_generator, preds)
