
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: multiplier for IQR (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def z_score_normalize(data, column):
    """
    Normalize data using Z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    normalized_data = data.copy()
    mean = normalized_data[column].mean()
    std = normalized_data[column].std()
    
    if std == 0:
        normalized_data[f'{column}_normalized'] = 0
    else:
        normalized_data[f'{column}_normalized'] = (normalized_data[column] - mean) / std
    
    return normalized_data

def min_max_normalize(data, column, feature_range=(0, 1)):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
        feature_range: tuple of (min, max) for output range
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    normalized_data = data.copy()
    min_val = normalized_data[column].min()
    max_val = normalized_data[column].max()
    
    if min_val == max_val:
        normalized_data[f'{column}_normalized'] = feature_range[0]
    else:
        normalized_data[f'{column}_normalized'] = (
            (normalized_data[column] - min_val) / (max_val - min_val) * 
            (feature_range[1] - feature_range[0]) + feature_range[0]
        )
    
    return normalized_data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        data: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: list of columns to process (None for all numeric columns)
    
    Returns:
        DataFrame with handled missing values
    """
    processed_data = data.copy()
    
    if columns is None:
        columns = processed_data.select_dtypes(include=[np.number]).columns
    
    for column in columns:
        if column not in processed_data.columns:
            continue
            
        if strategy == 'drop':
            processed_data = processed_data.dropna(subset=[column])
        elif strategy == 'mean':
            processed_data[column] = processed_data[column].fillna(processed_data[column].mean())
        elif strategy == 'median':
            processed_data[column] = processed_data[column].fillna(processed_data[column].median())
        elif strategy == 'mode':
            processed_data[column] = processed_data[column].fillna(processed_data[column].mode()[0])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return processed_data

def create_sample_data():
    """
    Create sample data for testing the cleaning functions.
    
    Returns:
        pandas DataFrame with sample data
    """
    np.random.seed(42)
    data = {
        'id': range(1, 101),
        'value': np.random.normal(100, 15, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'score': np.random.uniform(0, 100, 100)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[10:15, 'value'] = np.nan
    df.loc[95, 'value'] = 500
    df.loc[96, 'value'] = -200
    
    return df

if __name__ == "__main__":
    sample_data = create_sample_data()
    print("Original data shape:", sample_data.shape)
    
    cleaned_data = remove_outliers_iqr(sample_data, 'value')
    print("After outlier removal:", cleaned_data.shape)
    
    normalized_data = z_score_normalize(cleaned_data, 'value')
    print("After normalization:", normalized_data.shape)
    
    handled_data = handle_missing_values(normalized_data, strategy='mean')
    print("After handling missing values:", handled_data.shape)
    
    print("\nSample of processed data:")
    print(handled_data[['id', 'value', 'value_normalized']].head())import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a dataset by handling missing values and removing outliers.
    
    Args:
        df: pandas DataFrame
        missing_strategy: Strategy for handling missing values ('mean', 'median', 'drop')
        outlier_threshold: Z-score threshold for outlier detection
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean())
    elif missing_strategy == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median())
    elif missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    
    # Remove outliers using Z-score method
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs((cleaned_df[numeric_cols] - cleaned_df[numeric_cols].mean()) / 
                      cleaned_df[numeric_cols].std())
    
    outlier_mask = (z_scores < outlier_threshold).all(axis=1)
    cleaned_df = cleaned_df[outlier_mask]
    
    return cleaned_df.reset_index(drop=True)

def validate_data(df, required_columns=None, min_rows=10):
    """
    Validate dataset structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(df) < min_rows:
        return False, f"Dataset must have at least {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        inf_mask = np.isinf(df[numeric_cols]).any().any()
        if inf_mask:
            return False, "Dataset contains infinite values"
    
    return True, "Dataset validation passed"

def normalize_data(df, method='minmax'):
    """
    Normalize numeric columns in the dataset.
    
    Args:
        df: pandas DataFrame
        method: Normalization method ('minmax' or 'standard')
    
    Returns:
        Normalized pandas DataFrame
    """
    normalized_df = df.copy()
    numeric_cols = normalized_df.select_dtypes(include=[np.number]).columns
    
    if method == 'minmax':
        for col in numeric_cols:
            col_min = normalized_df[col].min()
            col_max = normalized_df[col].max()
            if col_max != col_min:
                normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
    
    elif method == 'standard':
        for col in numeric_cols:
            col_mean = normalized_df[col].mean()
            col_std = normalized_df[col].std()
            if col_std > 0:
                normalized_df[col] = (normalized_df[col] - col_mean) / col_std
    
    return normalized_df

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'feature1': [1, 2, np.nan, 4, 5, 100],
        'feature2': [10, 20, 30, np.nan, 50, 60],
        'category': ['A', 'B', 'A', 'B', 'A', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n")
    
    # Clean the data
    cleaned = clean_dataset(df, missing_strategy='mean', outlier_threshold=2)
    print("Cleaned dataset:")
    print(cleaned)
    print("\n")
    
    # Validate the cleaned data
    is_valid, message = validate_data(cleaned, min_rows=3)
    print(f"Validation: {is_valid} - {message}")
    print("\n")
    
    # Normalize the data
    normalized = normalize_data(cleaned, method='minmax')
    print("Normalized dataset:")
    print(normalized)