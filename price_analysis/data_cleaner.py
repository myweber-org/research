
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    cleaned_df = df.copy()

    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Dropped {original_shape[0] - cleaned_df.shape[0]} duplicate rows.")

    if fill_missing:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        if fill_missing == 'mean':
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
        elif fill_missing == 'median':
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
        elif fill_missing == 'zero':
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(0)
        print(f"Filled missing values in numeric columns using '{fill_missing}' strategy.")

    categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
    cleaned_df[categorical_cols] = cleaned_df[categorical_cols].fillna('Unknown')
    print("Filled missing values in categorical columns with 'Unknown'.")

    print(f"Original shape: {original_shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate the DataFrame for required columns and data types.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    if df.empty:
        raise ValueError("DataFrame is empty.")

    print("DataFrame validation passed.")
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, np.nan, 15.0, 20.0, np.nan, 30.0],
        'category': ['A', 'B', 'B', None, 'C', 'A']
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)

    cleaned_df = clean_dataframe(df, fill_missing='median')
    print("\nCleaned DataFrame:")
    print(cleaned_df)

    try:
        validate_dataframe(cleaned_df, required_columns=['id', 'value'])
    except ValueError as e:
        print(f"Validation error: {e}")