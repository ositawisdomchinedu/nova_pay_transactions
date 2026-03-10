import pandas as pd

DATA_PATH = "nova_pay_transcations (1).csv"

def load_data(file_path: str) -> pd.DataFrame:
    """Load transaction dataset."""
    return pd.read_csv(DATA_PATH)