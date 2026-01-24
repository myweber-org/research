
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using Interquartile Range method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

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

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization
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
    Handle missing values using specified strategy
    """
    if columns is None:
        columns = data.columns
    
    data_clean = data.copy()
    
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
            elif strategy == 'drop':
                data_clean = data_clean.dropna(subset=[column])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data_clean[column] = data_clean[column].fillna(fill_value)
    
    return data_clean

def validate_dataframe(data, required_columns=None, numeric_columns=None):
    """
    Validate DataFrame structure and content
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if numeric_columns:
        for column in numeric_columns:
            if column in data.columns:
                if not pd.api.types.is_numeric_dtype(data[column]):
                    raise ValueError(f"Column '{column}' must be numeric")
    
    return True

def create_sample_data():
    """
    Create sample data for testing
    """
    np.random.seed(42)
    n_samples = 100
    
    data = pd.DataFrame({
        'feature_a': np.random.normal(50, 15, n_samples),
        'feature_b': np.random.exponential(2, n_samples),
        'feature_c': np.random.uniform(0, 100, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples)
    })
    
    # Add some outliers
    data.loc[10, 'feature_a'] = 200
    data.loc[20, 'feature_b'] = 50
    
    # Add some missing values
    data.loc[30:35, 'feature_c'] = np.nan
    
    return data

if __name__ == "__main__":
    # Example usage
    sample_data = create_sample_data()
    print("Original data shape:", sample_data.shape)
    
    # Remove outliers
    cleaned_data, removed = remove_outliers_iqr(sample_data, 'feature_a')
    print(f"Removed {removed} outliers from feature_a")
    
    # Handle missing values
    filled_data = handle_missing_values(cleaned_data, strategy='mean')
    print("Data after handling missing values:", filled_data.shape)
    
    # Normalize features
    filled_data['feature_a_normalized'] = normalize_minmax(filled_data, 'feature_a')
    filled_data['feature_b_standardized'] = standardize_zscore(filled_data, 'feature_b')
    
    print("\nData cleaning completed successfully")
    print("Available columns:", filled_data.columns.tolist())