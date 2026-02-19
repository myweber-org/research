
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers using IQR method
    """
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    return filtered_df

def z_score_normalization(dataframe, column):
    """
    Apply z-score normalization to specified column
    """
    mean_val = dataframe[column].mean()
    std_val = dataframe[column].std()
    
    if std_val > 0:
        dataframe[column + '_normalized'] = (dataframe[column] - mean_val) / std_val
    else:
        dataframe[column + '_normalized'] = 0
    
    return dataframe

def min_max_scaling(dataframe, column, feature_range=(0, 1)):
    """
    Apply min-max scaling to specified column
    """
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    
    if max_val > min_val:
        scaled = (dataframe[column] - min_val) / (max_val - min_val)
        scaled = scaled * (feature_range[1] - feature_range[0]) + feature_range[0]
    else:
        scaled = feature_range[0]
    
    dataframe[column + '_scaled'] = scaled
    return dataframe

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values with specified strategy
    """
    df_copy = dataframe.copy()
    
    if columns is None:
        columns = df_copy.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if df_copy[col].isnull().any():
            if strategy == 'mean':
                fill_value = df_copy[col].mean()
            elif strategy == 'median':
                fill_value = df_copy[col].median()
            elif strategy == 'mode':
                fill_value = df_copy[col].mode()[0]
            elif strategy == 'zero':
                fill_value = 0
            else:
                raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', or 'zero'")
            
            df_copy[col] = df_copy[col].fillna(fill_value)
    
    return df_copy

def validate_dataframe(dataframe, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content
    """
    if len(dataframe) < min_rows:
        raise ValueError(f"DataFrame must have at least {min_rows} rows")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

def create_sample_data():
    """
    Create sample data for testing
    """
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.uniform(0, 1, 100),
        'feature_c': np.random.exponential(2, 100)
    }
    
    # Add some outliers
    data['feature_a'][95] = 300
    data['feature_a'][96] = -50
    
    # Add some missing values
    data['feature_b'][10] = np.nan
    data['feature_b'][20] = np.nan
    
    return pd.DataFrame(data)
def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    Returns a new list with unique elements.
    """
    seen = set()
    unique_list = []
    for item in data_list:
        if item not in seen:
            seen.add(item)
            unique_list.append(item)
    return unique_list

def clean_numeric_data(values, default=0):
    """
    Clean a list of numeric values by converting non-numeric entries to default.
    Returns a list of cleaned numeric values.
    """
    cleaned = []
    for val in values:
        try:
            cleaned.append(float(val))
        except (ValueError, TypeError):
            cleaned.append(default)
    return cleaned

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 5]
    print("Original:", sample_data)
    print("Cleaned:", remove_duplicates(sample_data))
    
    mixed_data = [1, "2", "abc", 3.5, None]
    print("Numeric cleaned:", clean_numeric_data(mixed_data))