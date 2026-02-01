
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df

def z_score_normalize(dataframe, columns):
    """
    Apply z-score normalization to specified columns
    """
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col not in normalized_df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
        
        mean_val = normalized_df[col].mean()
        std_val = normalized_df[col].std()
        
        if std_val > 0:
            normalized_df[col] = (normalized_df[col] - mean_val) / std_val
        else:
            normalized_df[col] = 0
    
    return normalized_df

def min_max_normalize(dataframe, columns, feature_range=(0, 1)):
    """
    Apply min-max normalization to specified columns
    """
    normalized_df = dataframe.copy()
    min_val, max_val = feature_range
    
    for col in columns:
        if col not in normalized_df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
        
        col_min = normalized_df[col].min()
        col_max = normalized_df[col].max()
        
        if col_max > col_min:
            normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
            normalized_df[col] = normalized_df[col] * (max_val - min_val) + min_val
        else:
            normalized_df[col] = min_val
    
    return normalized_df

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy
    """
    processed_df = dataframe.copy()
    
    if columns is None:
        columns = processed_df.columns
    
    for col in columns:
        if col not in processed_df.columns:
            continue
            
        if processed_df[col].isnull().any():
            if strategy == 'mean':
                fill_value = processed_df[col].mean()
            elif strategy == 'median':
                fill_value = processed_df[col].median()
            elif strategy == 'mode':
                fill_value = processed_df[col].mode()[0]
            elif strategy == 'drop':
                processed_df = processed_df.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            processed_df[col] = processed_df[col].fillna(fill_value)
    
    return processed_df

def create_data_summary(dataframe):
    """
    Create comprehensive data summary
    """
    summary = {
        'total_rows': len(dataframe),
        'total_columns': len(dataframe.columns),
        'missing_values': dataframe.isnull().sum().sum(),
        'duplicate_rows': dataframe.duplicated().sum(),
        'numeric_columns': list(dataframe.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(dataframe.select_dtypes(include=['object']).columns),
        'column_types': dataframe.dtypes.to_dict()
    }
    
    return summary