
import pandas as pd
import numpy as np
from typing import List, Optional

def remove_duplicates(
    data: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first',
    inplace: bool = False
) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    data: Input DataFrame
    subset: Column labels to consider for identifying duplicates
    keep: Which duplicates to keep - 'first', 'last', or False
    inplace: Whether to modify the DataFrame in place
    
    Returns:
    DataFrame with duplicates removed
    """
    if not inplace:
        data = data.copy()
    
    if subset is None:
        subset = data.columns.tolist()
    
    cleaned_data = data.drop_duplicates(subset=subset, keep=keep)
    
    if inplace:
        data.drop(data.index, inplace=True)
        data.update(cleaned_data)
        return data
    
    return cleaned_data

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df: DataFrame to validate
    
    Returns:
    Boolean indicating if DataFrame is valid
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if df.empty:
        return True
    
    if df.isnull().all().all():
        return False
    
    return True

def clean_numeric_columns(
    df: pd.DataFrame,
    columns: List[str],
    fill_method: str = 'mean'
) -> pd.DataFrame:
    """
    Clean numeric columns by handling missing values.
    
    Parameters:
    df: Input DataFrame
    columns: List of numeric column names to clean
    fill_method: Method to fill missing values - 'mean', 'median', or 'zero'
    
    Returns:
    DataFrame with cleaned numeric columns
    """
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
        
        if fill_method == 'mean':
            fill_value = df_clean[col].mean()
        elif fill_method == 'median':
            fill_value = df_clean[col].median()
        elif fill_method == 'zero':
            fill_value = 0
        else:
            fill_value = df_clean[col].mean()
        
        df_clean[col] = df_clean[col].fillna(fill_value)
    
    return df_clean

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'id': [1, 2, 2, 3, 4, 4],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David'],
        'score': [85, 90, 90, 78, None, 92],
        'age': [25, 30, 30, 22, 35, 35]
    })
    
    print("Original data:")
    print(sample_data)
    print("\nAfter removing duplicates:")
    cleaned = remove_duplicates(sample_data, subset=['id', 'name'])
    print(cleaned)
    
    print("\nAfter cleaning numeric columns:")
    numeric_cleaned = clean_numeric_columns(cleaned, columns=['score'], fill_method='mean')
    print(numeric_cleaned)
import numpy as np

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_summary_statistics(data, column):
    mean_val = data[column].mean()
    median_val = data[column].median()
    std_val = data[column].std()
    return {'mean': mean_val, 'median': median_val, 'std': std_val}import pandas as pd
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
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[column]):
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Warning: Could not clean column '{column}': {e}")
    
    return cleaned_df

def save_cleaned_data(df, input_path, output_suffix="_cleaned"):
    """
    Save cleaned DataFrame to a new CSV file.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame
        input_path (str): Original file path
        output_suffix (str): Suffix to add to output filename
    """
    if input_path.endswith('.csv'):
        output_path = input_path.replace('.csv', f'{output_suffix}.csv')
    else:
        output_path = f"{input_path}{output_suffix}.csv"
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': range(1, 101),
        'value': np.concatenate([
            np.random.normal(100, 10, 90),
            np.random.normal(300, 50, 10)  # Outliers
        ]),
        'score': np.concatenate([
            np.random.normal(50, 5, 95),
            np.random.normal(100, 10, 5)  # Outliers
        ])
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Original data shape: {df.shape}")
    
    cleaned_df = clean_numeric_data(df, columns=['value', 'score'])
    print(f"Cleaned data shape: {cleaned_df.shape}")
    
    print("Outlier removal complete.")import pandas as pd
import numpy as np

def load_data(filepath):
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def remove_outliers_iqr(df, column):
    """Remove outliers from a column using the IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    removed_count = len(df) - len(filtered_df)
    print(f"Removed {removed_count} outliers from column '{column}'")
    return filtered_df

def normalize_column(df, column):
    """Normalize a column using min-max scaling."""
    if column in df.columns:
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val != min_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
            print(f"Column '{column}' normalized to range [0, 1]")
        else:
            print(f"Column '{column}' has constant values, skipping normalization")
    else:
        print(f"Column '{column}' not found in DataFrame")
    return df

def clean_data(df, numeric_columns):
    """Main data cleaning function."""
    if df is None or df.empty:
        print("No data to clean")
        return df
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
    
    for col in numeric_columns:
        if col in df.columns:
            df = normalize_column(df, col)
    
    print(f"Cleaned data shape: {df.shape}")
    print(f"Rows removed: {original_shape[0] - df.shape[0]}")
    return df

def save_cleaned_data(df, output_path):
    """Save cleaned data to CSV."""
    if df is not None and not df.empty:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
    else:
        print("No data to save")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    numeric_cols = ["age", "income", "score"]
    
    raw_data = load_data(input_file)
    if raw_data is not None:
        cleaned_data = clean_data(raw_data, numeric_cols)
        save_cleaned_data(cleaned_data, output_file)