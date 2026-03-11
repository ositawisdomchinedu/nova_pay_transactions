from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd()

DATA_PATH = PROJECT_ROOT / "data" / "nova_pay_transaction.csv"

def load_data(file_path: str) -> pd.DataFrame:
    """Load raw transaction dataset."""
    path = Path(file_path)
    return pd.read_csv(path)

