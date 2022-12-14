import numpy as np
import pandas as pd
from pathlib import Path


def load_disease():
    file_path = Path(__file__).parent / "data" / "disease.csv"
    df = pd.read_csv(file_path, dtype=np.float_)
    return df
