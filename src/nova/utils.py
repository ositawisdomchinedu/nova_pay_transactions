import joblib
import os


def save_object(obj, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(obj, file_path)


def load_object(file_path: str):
    return joblib.load(file_path)