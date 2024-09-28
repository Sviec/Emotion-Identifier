import pickle
import tensorflow as tf
from src.model.modeling.fit_model import train_model, fit_model, fit_fine_model
from src.visualization.evaluate import plot_training_history
from src.model.preprocessing.hyperopt import run_optimization
from config import EFFICIENTNET_DIR, EFFICIENTNET_FINE_DIR, HISTORY_DIR
from src.data.load_data import load_data


def train(model_name):
    model, train_generator, validation_generator = train_model()
    model, history = fit_model(model_name, model, train_generator, validation_generator)

    plot_training_history(history)

    model.save(EFFICIENTNET_DIR)

    with open(f'{HISTORY_DIR}/history.pkl', 'wb') as f:
        pickle.dump(history.history, f)


def train_fine(model_name):
    train_generator, validation_generator = load_data()
    model = tf.keras.models.load_model(f'{EFFICIENTNET_DIR}')
    with open(f'{HISTORY_DIR}/history.pkl', 'rb') as f:
        history = pickle.load(f)

    model_fine, history_fine = fit_fine_model(
        model_name, model, history,
        train_generator, validation_generator
    )
    plot_training_history(history_fine, True)

    model.save(EFFICIENTNET_FINE_DIR)