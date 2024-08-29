from src.models.train_model import train_model, build_model, fit_model
from src.models.evaluate import plot_training_history

if __name__ == '__main__':

    for model_name in ["efficientnet", "convnextsmall"]:
        model, train_generator, validation_generator, total_epochs, initial_epochs = train_model(model_name)
        model, history = fit_model(model_name, model, train_generator, validation_generator, initial_epochs)

        plot_training_history(history)

        model.save_weights(f'results/{model_name}/emotion_model.h5')
        with open(f'results/{model_name}/emotion_model.json', 'w') as json_file:
            json_file.write(model.to_json())

        model_fine, history_fine = fit_model(
            model_name, model,
            train_generator, validation_generator,
            total_epochs, 0.00001, True
        )
        plot_training_history(history_fine)

        model_fine.save_weights(f'results/{model_name}/emotion_fine_model.h5')
        with open(f'results/{model_name}/emotion_fine_model.json', 'w') as json_file:
            json_file.write(model_fine.to_json())
