import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of column names to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            except Exception as e:
                print(f"Warning: Could not process column '{col}': {e}")
    
    return cleaned_df

def save_cleaned_data(df, input_path, suffix='_cleaned'):
    """
    Save cleaned DataFrame to CSV with modified filename.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame
        input_path (str): Original file path
        suffix (str): Suffix to add to filename
    
    Returns:
        str: Path to saved file
    """
    if not input_path.endswith('.csv'):
        raise ValueError("Input path must be a CSV file")
    
    output_path = input_path.replace('.csv', f'{suffix}.csv')
    df.to_csv(output_path, index=False)
    
    return output_path

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 21),
        'value': [10, 12, 11, 15, 9, 100, 13, 14, 12, 11,
                  10, 9, 8, 12, 13, 200, 14, 15, 11, 10]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data shape:", df.shape)
    print("Original data summary:")
    print(df['value'].describe())
    
    cleaned_df = clean_numeric_data(df, columns=['value'])
    print("\nCleaned data shape:", cleaned_df.shape)
    print("Cleaned data summary:")
    print(cleaned_df['value'].describe())