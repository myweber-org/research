import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    print(f"Original dataset shape: {original_shape}")

    if drop_duplicates:
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows.")

    if fill_missing:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if fill_missing == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif fill_missing == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif fill_missing == 'zero':
            df[numeric_cols] = df[numeric_cols].fillna(0)
        print(f"Filled missing values in numeric columns using method: {fill_missing}")

    print(f"Cleaned dataset shape: {df.shape}")
    return df

def validate_data(df, required_columns=None):
    """
    Validate the dataset for required columns and basic integrity.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    if df.isnull().sum().sum() > 0:
        print("Warning: Dataset still contains missing values.")

    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, 5, np.nan],
        'B': [10, 20, 20, 40, 50, 60],
        'C': [100, 200, 300, np.nan, 500, 600]
    }
    df = pd.DataFrame(sample_data)
    cleaned_df = clean_dataset(df, fill_missing='mean')
    validate_data(cleaned_df, required_columns=['A', 'B', 'C'])