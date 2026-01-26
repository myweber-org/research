
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
    print(handled_data[['id', 'value', 'value_normalized']].head())