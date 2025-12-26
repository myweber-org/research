
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    """
    Remove outliers using IQR method for specified columns.
    Returns cleaned DataFrame and outlier indices.
    """
    clean_df = df.copy()
    outlier_indices = []
    
    for col in columns:
        if col not in clean_df.columns:
            continue
            
        Q1 = clean_df[col].quantile(0.25)
        Q3 = clean_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        col_outliers = clean_df[(clean_df[col] < lower_bound) | (clean_df[col] > upper_bound)].index
        outlier_indices.extend(col_outliers)
        clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]
    
    return clean_df, list(set(outlier_indices))

def remove_outliers_zscore(df, columns, threshold=3):
    """
    Remove outliers using Z-score method.
    Returns cleaned DataFrame and outlier indices.
    """
    clean_df = df.copy()
    outlier_indices = []
    
    for col in columns:
        if col not in clean_df.columns:
            continue
            
        z_scores = np.abs(stats.zscore(clean_df[col].dropna()))
        col_outliers = clean_df[z_scores > threshold].index
        outlier_indices.extend(col_outliers)
        clean_df = clean_df[z_scores <= threshold]
    
    return clean_df, list(set(outlier_indices))

def normalize_minmax(df, columns):
    """
    Apply Min-Max normalization to specified columns.
    Returns DataFrame with normalized columns.
    """
    normalized_df = df.copy()
    
    for col in columns:
        if col not in normalized_df.columns:
            continue
            
        min_val = normalized_df[col].min()
        max_val = normalized_df[col].max()
        
        if max_val != min_val:
            normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
        else:
            normalized_df[col] = 0
    
    return normalized_df

def normalize_zscore(df, columns):
    """
    Apply Z-score normalization to specified columns.
    Returns DataFrame with normalized columns.
    """
    normalized_df = df.copy()
    
    for col in columns:
        if col not in normalized_df.columns:
            continue
            
        mean_val = normalized_df[col].mean()
        std_val = normalized_df[col].std()
        
        if std_val != 0:
            normalized_df[col] = (normalized_df[col] - mean_val) / std_val
        else:
            normalized_df[col] = 0
    
    return normalized_df

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    Supported strategies: 'mean', 'median', 'mode', 'drop'
    """
    if columns is None:
        columns = df.columns
    
    processed_df = df.copy()
    
    if strategy == 'drop':
        return processed_df.dropna(subset=columns)
    
    for col in columns:
        if col not in processed_df.columns:
            continue
            
        if strategy == 'mean':
            fill_value = processed_df[col].mean()
        elif strategy == 'median':
            fill_value = processed_df[col].median()
        elif strategy == 'mode':
            fill_value = processed_df[col].mode()[0] if not processed_df[col].mode().empty else 0
        else:
            fill_value = 0
        
        processed_df[col] = processed_df[col].fillna(fill_value)
    
    return processed_df

def clean_dataset(df, numerical_columns, categorical_columns=None, 
                  outlier_method='iqr', normalize_method='minmax',
                  missing_strategy='mean'):
    """
    Complete data cleaning pipeline.
    """
    if categorical_columns is None:
        categorical_columns = []
    
    all_columns = numerical_columns + categorical_columns
    
    print(f"Initial dataset shape: {df.shape}")
    
    cleaned_df = handle_missing_values(df, strategy=missing_strategy, columns=all_columns)
    print(f"After handling missing values: {cleaned_df.shape}")
    
    if outlier_method == 'iqr':
        cleaned_df, outliers = remove_outliers_iqr(cleaned_df, numerical_columns)
    elif outlier_method == 'zscore':
        cleaned_df, outliers = remove_outliers_zscore(cleaned_df, numerical_columns)
    else:
        outliers = []
    
    print(f"Outliers removed: {len(outliers)}")
    print(f"After outlier removal: {cleaned_df.shape}")
    
    if normalize_method == 'minmax':
        cleaned_df = normalize_minmax(cleaned_df, numerical_columns)
    elif normalize_method == 'zscore':
        cleaned_df = normalize_zscore(cleaned_df, numerical_columns)
    
    print(f"Final dataset shape: {cleaned_df.shape}")
    
    return cleaned_df, outliers

if __name__ == "__main__":
    sample_data = {
        'feature1': [1, 2, 3, 4, 100, 6, 7, 8, 9, 10],
        'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 1000],
        'feature3': [100, 200, 300, 400, 500, 600, 700, 800, 900, 10000],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    numerical_cols = ['feature1', 'feature2', 'feature3']
    categorical_cols = ['category']
    
    cleaned_data, removed_outliers = clean_dataset(
        df, 
        numerical_columns=numerical_cols,
        categorical_columns=categorical_cols,
        outlier_method='iqr',
        normalize_method='minmax',
        missing_strategy='mean'
    )
    
    print("\nCleaned Data Summary:")
    print(cleaned_data.describe())