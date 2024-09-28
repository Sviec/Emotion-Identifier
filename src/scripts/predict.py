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
    return predictions.argmax(axis=1)


def indices_to_labels(predictions: np.array):
    to_indices = np.vectorize(lambda i: INDICES_TO_LABELS[i])
    return to_indices(predictions)


def predictions_to_labels(predictions: np.array):
    indices = predictions_to_indices(predictions)
    labels = indices_to_labels(indices)
    return labels


def get_predictions(model_name, model, test_generator):
    probs = model.predict(test_generator, verbose=1)
    np.save(f'../results/{model_name}/predictions/predictions.npy', probs)
    probs = np.load(f'../results/{model_name}/predictions/predictions.npy')
    preds = predictions_to_labels(probs)

    return preds


