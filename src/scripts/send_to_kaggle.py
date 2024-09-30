import pandas as pd
from config import SOLUTION


def get_filename_from_path(path_to_file: str):
    return path_to_file.split('\\')[-1]


def send_to_kaggle(test_generator, preds):
    submission = pd.DataFrame({
        'image_path': test_generator.file_paths,
        'emotion': preds
    })
    submission['image_path'] = submission['image_path'].apply(get_filename_from_path)
    print(submission.head())
    submission.to_csv(f'{SOLUTION}/submission.csv', index=False)