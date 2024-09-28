import matplotlib.pyplot as plt
from config import VISUAL


def plot_training_history(history, fine=False):
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Categorical Accuracy')
    plt.plot(val_acc, label='Validation Categorical Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Categorical Accuracy')
    plt.ylim([0, 1])
    plt.title('Training and Validation Categorical Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 3.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    if fine:
        plt.savefig(f'{VISUAL}/training_fine_plot.png')
    else:
        plt.savefig(f'{VISUAL}/training_plot.png')
