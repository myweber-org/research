import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers from a column using the IQR method.
    
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
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    """
    Normalize a column using min-max scaling to [0, 1] range.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        data[column + '_normalized'] = 0.5
    else:
        data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    
    return data

def standardize_zscore(data, column):
    """
    Standardize a column using z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        DataFrame with standardized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        data[column + '_standardized'] = 0
    else:
        data[column + '_standardized'] = (data[column] - mean_val) / std_val
    
    return data

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Args:
        data: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        DataFrame with handled missing values
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return data.dropna(subset=numeric_cols)
    
    for col in numeric_cols:
        if data[col].isnull().any():
            if strategy == 'mean':
                fill_value = data[col].mean()
            elif strategy == 'median':
                fill_value = data[col].median()
            elif strategy == 'mode':
                fill_value = data[col].mode()[0]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data[col] = data[col].fillna(fill_value)
    
    return data

def create_sample_data():
    """
    Create sample data for testing the cleaning functions.
    
    Returns:
        pandas DataFrame with sample data
    """
    np.random.seed(42)
    n_samples = 100
    
    data = pd.DataFrame({
        'feature_a': np.random.normal(50, 15, n_samples),
        'feature_b': np.random.exponential(10, n_samples),
        'feature_c': np.random.uniform(0, 100, n_samples)
    })
    
    # Add some outliers
    data.loc[0, 'feature_a'] = 200
    data.loc[1, 'feature_b'] = 150
    
    # Add some missing values
    data.loc[5:10, 'feature_c'] = np.nan
    
    return data

if __name__ == "__main__":
    # Example usage
    sample_data = create_sample_data()
    print("Original data shape:", sample_data.shape)
    print("Original data head:\n", sample_data.head())
    
    # Remove outliers
    cleaned_data = remove_outliers_iqr(sample_data, 'feature_a')
    print("\nAfter outlier removal shape:", cleaned_data.shape)
    
    # Handle missing values
    filled_data = handle_missing_values(sample_data, strategy='mean')
    print("\nAfter handling missing values:")
    print("Missing values in feature_c:", filled_data['feature_c'].isnull().sum())
    
    # Normalize data
    normalized_data = normalize_minmax(filled_data, 'feature_b')
    print("\nNormalized feature_b range:", 
          normalized_data['feature_b_normalized'].min(), 
          "to", 
          normalized_data['feature_b_normalized'].max())