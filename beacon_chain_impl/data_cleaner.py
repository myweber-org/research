import re
from typing import List, Optional

def remove_special_characters(text: str, keep_spaces: bool = True) -> str:
    """
    Remove all non-alphanumeric characters from the input string.
    Optionally preserve spaces.
    """
    if keep_spaces:
        pattern = r'[^A-Za-z0-9\s]'
    else:
        pattern = r'[^A-Za-z0-9]'
    return re.sub(pattern, '', text)

def normalize_whitespace(text: str) -> str:
    """
    Replace multiple whitespace characters with a single space.
    Also strip leading and trailing whitespace.
    """
    return ' '.join(text.split())

def tokenize_text(text: str, lowercase: bool = True) -> List[str]:
    """
    Split text into tokens (words). Optionally convert to lowercase.
    """
    if lowercase:
        text = text.lower()
    tokens = text.split()
    return tokens

def clean_text_pipeline(
    text: str,
    remove_special: bool = True,
    normalize_ws: bool = True,
    tokenize: bool = False
) -> Optional[str]:
    """
    Apply a series of cleaning operations to the input text.
    Returns cleaned string or token list based on parameters.
    """
    if not isinstance(text, str):
        return None

    cleaned = text

    if remove_special:
        cleaned = remove_special_characters(cleaned)

    if normalize_ws:
        cleaned = normalize_whitespace(cleaned)

    if tokenize:
        return tokenize_text(cleaned)

    return cleaned

def batch_clean_texts(texts: List[str], **kwargs) -> List[Optional[str]]:
    """
    Apply cleaning pipeline to a list of text strings.
    Returns list of cleaned texts (or None for invalid inputs).
    """
    return [clean_text_pipeline(text, **kwargs) for text in texts]
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to clean.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(file_path, output_path=None):
    """
    Load a dataset, clean specified columns, and optionally save the result.
    
    Parameters:
    file_path (str): Path to the input CSV file.
    output_path (str, optional): Path to save the cleaned CSV file.
    
    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    df = pd.read_csv(file_path)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_data = clean_dataset(input_file, output_file)
        print(f"Original data shape: {pd.read_csv(input_file).shape}")
        print(f"Cleaned data shape: {cleaned_data.shape}")
        print("Data cleaning completed successfully.")
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, columns):
    """Remove outliers using IQR method."""
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def normalize_data(df, columns):
    """Normalize data using min-max scaling."""
    df_normalized = df.copy()
    for col in columns:
        min_val = df_normalized[col].min()
        max_val = df_normalized[col].max()
        df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    return df_normalized

def handle_missing_values(df, strategy='mean'):
    """Handle missing values with specified strategy."""
    df_filled = df.copy()
    numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            df_filled[col].fillna(df_filled[col].mean(), inplace=True)
    elif strategy == 'median':
        for col in numeric_cols:
            df_filled[col].fillna(df_filled[col].median(), inplace=True)
    elif strategy == 'mode':
        for col in numeric_cols:
            df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)
    
    return df_filled

def clean_dataset(filepath, numeric_columns):
    """Main function to clean dataset."""
    df = load_dataset(filepath)
    df = handle_missing_values(df, strategy='median')
    df = remove_outliers_iqr(df, numeric_columns)
    df = normalize_data(df, numeric_columns)
    return df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 100, 7, 8, 9, 10],
        'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'feature3': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_dataset('sample_data.csv', ['feature1', 'feature2', 'feature3'])
    print("Cleaned dataset shape:", cleaned_df.shape)
    print("\nCleaned data summary:")
    print(cleaned_df.describe())import numpy as np
import pandas as pd

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
    if max_val == min_val:
        return df[column].apply(lambda x: 0.0)
    return df[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_data(df, required_columns):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return True
import pandas as pd

def clean_dataset(df, subset=None, fill_method='mean'):
    """
    Cleans a pandas DataFrame by removing duplicates and handling missing values.

    Args:
        df (pd.DataFrame): The input DataFrame to clean.
        subset (list, optional): Column labels to consider for identifying duplicates.
                                 If None, all columns are used.
        fill_method (str, optional): Method to fill missing values.
                                     Options: 'mean', 'median', 'mode', or a constant value.
                                     Defaults to 'mean' for numeric columns.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()

    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates(subset=subset, keep='first')

    # Handle missing values
    for column in cleaned_df.columns:
        if cleaned_df[column].isnull().any():
            if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                if fill_method == 'mean':
                    fill_value = cleaned_df[column].mean()
                elif fill_method == 'median':
                    fill_value = cleaned_df[column].median()
                elif fill_method == 'mode':
                    fill_value = cleaned_df[column].mode()[0]
                else:
                    try:
                        fill_value = float(fill_method)
                    except ValueError:
                        raise ValueError(f"Invalid fill_method for numeric column '{column}': {fill_method}")
                cleaned_df[column].fillna(fill_value, inplace=True)
            else:
                # For non-numeric columns, fill with the mode (most frequent value)
                fill_value = cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else 'Unknown'
                cleaned_df[column].fillna(fill_value, inplace=True)

    return cleaned_dfimport numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers from specified column using IQR method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize column values to range [0, 1] using min-max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize column values using z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_factor=1.5):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric)
        outlier_factor: IQR factor for outlier removal
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, column, outlier_factor)
            cleaned_data[column + '_normalized'] = normalize_minmax(cleaned_data, column)
            cleaned_data[column + '_standardized'] = standardize_zscore(cleaned_data, column)
    
    return cleaned_data

def validate_data(data, required_columns):
    """
    Validate that DataFrame contains required columns and no null values.
    
    Args:
        data: pandas DataFrame
        required_columns: list of required column names
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    null_counts = data[required_columns].isnull().sum()
    if null_counts.any():
        return False, f"Null values found in columns: {null_counts[null_counts > 0].to_dict()}"
    
    return True, "Data validation passed"
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from the specified column.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_summary_statistics(data, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing count, mean, std, min, and max.
    """
    stats = {
        'count': data[column].count(),
        'mean': data[column].mean(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max()
    }
    return stats