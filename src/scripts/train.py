from src.models.train_model import train_model, fit_model
from src.models.evaluate import plot_training_history


def train(model_name):
    model, train_generator, validation_generator, total_epochs, initial_epochs = train_model(model_name)
    model, history = fit_model(model_name, model, train_generator, validation_generator, initial_epochs)

    plot_training_history(history)

    model.save(f'../results/{model_name}/emotion_model.keras')

    model_fine, history_fine = fit_model(
        model_name, model,
        train_generator, validation_generator,
        total_epochs, 0.0001, True
    )
    plot_training_history(history_fine)

    model.save(f'results/{model_name}/emotion_fine_model.keras')

