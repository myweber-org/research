
import pandas as pd

def clean_dataset(df, columns=None, drop_duplicates=True, fill_method=None):
    """
    Clean a pandas DataFrame by handling missing values and removing duplicates.
    
    Args:
        df: pandas DataFrame to clean
        columns: list of columns to clean, if None clean all columns
        drop_duplicates: boolean, whether to remove duplicate rows
        fill_method: method to fill missing values ('mean', 'median', 'mode', or None)
    
    Returns:
        Cleaned pandas DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.columns
    
    for col in columns:
        if col in df_clean.columns:
            if fill_method == 'mean' and pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            elif fill_method == 'median' and pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif fill_method == 'mode':
                df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else None, inplace=True)
    
    if drop_duplicates:
        df_clean.drop_duplicates(inplace=True)
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of required column names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(50, 200),
        'category': np.random.choice(['X', 'Y', 'Z'], 200)
    })
    
    numeric_cols = ['feature_a', 'feature_b']
    result = clean_dataset(sample_data, numeric_cols)
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {result.shape}")
    print(result.head())