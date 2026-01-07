
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
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

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns
    """
    if columns is None:
        columns = data.columns
    
    data_filled = data.copy()
    
    for column in columns:
        if column not in data.columns:
            continue
            
        if data[column].isnull().any():
            if strategy == 'mean':
                fill_value = data[column].mean()
            elif strategy == 'median':
                fill_value = data[column].median()
            elif strategy == 'mode':
                fill_value = data[column].mode()[0]
            elif strategy == 'constant':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data_filled[column] = data[column].fillna(fill_value)
    
    return data_filled

def clean_dataset(data, outlier_method='iqr', normalize_method='minmax', 
                  missing_strategy='mean', outlier_columns=None, normalize_columns=None):
    """
    Comprehensive data cleaning pipeline
    """
    cleaned_data = data.copy()
    
    if outlier_columns is None:
        outlier_columns = cleaned_data.select_dtypes(include=[np.number]).columns
    
    for column in outlier_columns:
        if column in cleaned_data.columns:
            if outlier_method == 'iqr':
                cleaned_data = remove_outliers_iqr(cleaned_data, column)
            elif outlier_method == 'zscore':
                cleaned_data = remove_outliers_zscore(cleaned_data, column)
    
    cleaned_data = handle_missing_values(cleaned_data, strategy=missing_strategy)
    
    if normalize_columns is None:
        normalize_columns = cleaned_data.select_dtypes(include=[np.number]).columns
    
    for column in normalize_columns:
        if column in cleaned_data.columns:
            if normalize_method == 'minmax':
                cleaned_data[column] = normalize_minmax(cleaned_data, column)
            elif normalize_method == 'zscore':
                cleaned_data[column] = normalize_zscore(cleaned_data, column)
    
    return cleaned_dataimport numpy as np
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
    return Trueimport pandas as pd
import numpy as np

def clean_missing_data(df, strategy='mean', columns=None):
    """
    Clean missing data in a pandas DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Strategy for handling missing values.
                        Options: 'mean', 'median', 'mode', 'drop', 'fill_zero'
        columns (list): List of columns to apply cleaning. If None, applies to all numeric columns.

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_cleaned = df.copy()
    
    if columns is None:
        columns = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in df_cleaned.columns:
            continue
            
        if strategy == 'mean':
            fill_value = df_cleaned[col].mean()
        elif strategy == 'median':
            fill_value = df_cleaned[col].median()
        elif strategy == 'mode':
            fill_value = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 0
        elif strategy == 'fill_zero':
            fill_value = 0
        elif strategy == 'drop':
            df_cleaned = df_cleaned.dropna(subset=[col])
            continue
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        df_cleaned[col] = df_cleaned[col].fillna(fill_value)
    
    return df_cleaned

def remove_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Remove outliers using the Interquartile Range method.

    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to check for outliers
        threshold (float): IQR multiplier threshold

    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_filtered = df.copy()
    
    for col in columns:
        if col not in df_filtered.columns:
            continue
            
        Q1 = df_filtered[col].quantile(0.25)
        Q3 = df_filtered[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        df_filtered = df_filtered[(df_filtered[col] >= lower_bound) & 
                                  (df_filtered[col] <= upper_bound)]
    
    return df_filtered

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize data in specified columns.

    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to normalize
        method (str): Normalization method ('minmax' or 'zscore')

    Returns:
        pd.DataFrame: DataFrame with normalized columns
    """
    df_normalized = df.copy()
    
    if columns is None:
        columns = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in df_normalized.columns:
            continue
            
        if method == 'minmax':
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            if max_val != min_val:
                df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean_val = df_normalized[col].mean()
            std_val = df_normalized[col].std()
            if std_val != 0:
                df_normalized[col] = (df_normalized[col] - mean_val) / std_val
    
    return df_normalized

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.

    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required

    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

def save_cleaned_data(df, output_path, index=False):
    """
    Save cleaned DataFrame to CSV file.

    Args:
        df (pd.DataFrame): DataFrame to save
        output_path (str): Path to output CSV file
        index (bool): Whether to save index
    """
    df.to_csv(output_path, index=index)
    print(f"Cleaned data saved to: {output_path}")
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        columns_to_check (list, optional): Specific columns to check for duplicates. 
                                          If None, checks all columns.
        fill_strategy (str): Strategy for filling missing values. 
                            Options: 'mean', 'median', 'mode', 'zero', 'drop'
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    original_shape = df.shape
    
    # Remove duplicates
    if columns_to_check:
        df_cleaned = df.drop_duplicates(subset=columns_to_check)
    else:
        df_cleaned = df.drop_duplicates()
    
    duplicates_removed = original_shape[0] - df_cleaned.shape[0]
    
    # Handle missing values
    if fill_strategy == 'drop':
        df_cleaned = df_cleaned.dropna()
    elif fill_strategy in ['mean', 'median']:
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_strategy == 'mean':
                fill_value = df_cleaned[col].mean()
            else:  # median
                fill_value = df_cleaned[col].median()
            df_cleaned[col] = df_cleaned[col].fillna(fill_value)
    elif fill_strategy == 'mode':
        for col in df_cleaned.columns:
            mode_value = df_cleaned[col].mode()
            if not mode_value.empty:
                df_cleaned[col] = df_cleaned[col].fillna(mode_value.iloc[0])
    elif fill_strategy == 'zero':
        df_cleaned = df_cleaned.fillna(0)
    
    missing_filled = df.isna().sum().sum() - df_cleaned.isna().sum().sum()
    
    print(f"Original shape: {original_shape}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Missing values filled: {missing_filled}")
    print(f"Cleaned shape: {df_cleaned.shape}")
    
    return df_cleaned

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of column names that must be present
        min_rows (int): Minimum number of rows required
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data with duplicates and missing values
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', None, 'Eve'],
        'age': [25, 30, 30, None, 35, 28],
        'score': [85.5, 92.0, 92.0, 78.5, None, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, columns_to_check=['id', 'name'], fill_strategy='mean')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    is_valid = validate_dataframe(cleaned_df, required_columns=['id', 'name', 'age', 'score'], min_rows=3)
    print(f"\nData validation passed: {is_valid}")