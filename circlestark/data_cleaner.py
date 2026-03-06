
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_data = data[(z_scores < threshold) | (data[column].isna())]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_data(data, column):
    """
    Standardize data using Z-score normalization
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns=None, outlier_method='zscore', normalize=False):
    """
    Main function to clean dataset with multiple options
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column not in cleaned_df.columns:
            continue
            
        # Handle missing values
        if cleaned_df[column].isna().any():
            cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
        
        # Remove outliers
        if outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, column)
        elif outlier_method == 'iqr':
            outliers, _, _ = detect_outliers_iqr(cleaned_df, column)
            cleaned_df = cleaned_df[~cleaned_df.index.isin(outliers.index)]
        
        # Normalize if requested
        if normalize:
            cleaned_df[f'{column}_normalized'] = normalize_minmax(cleaned_df, column)
            cleaned_df[f'{column}_standardized'] = standardize_data(cleaned_df, column)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Add some outliers
    df.loc[10, 'feature1'] = 500
    df.loc[20, 'feature2'] = 1000
    
    # Clean the data
    cleaned = clean_dataset(df, numeric_columns=['feature1', 'feature2'], 
                           outlier_method='zscore', normalize=True)
    
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    print(f"Removed {len(df) - len(cleaned)} outliers")