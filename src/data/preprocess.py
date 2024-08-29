import cv2
import numpy as np

def preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize to [0, 1]
    return np.expand_dims(image, axis=0)
