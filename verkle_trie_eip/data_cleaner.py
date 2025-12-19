
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    cleaned_df = cleaned_df.dropna()
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 200),
        'B': np.random.exponential(50, 200),
        'C': np.random.uniform(0, 1, 200)
    })
    sample_data.loc[10, 'A'] = 500
    sample_data.loc[20, 'B'] = 1000
    cleaned = clean_dataset(sample_data, ['A', 'B', 'C'])
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def calculate_statistics(df, column):
    mean_val = df[column].mean()
    median_val = df[column].median()
    std_val = df[column].std()
    return {
        'mean': mean_val,
        'median': median_val,
        'std': std_val
    }

def process_dataframe(input_df, target_column):
    cleaned_df = remove_outliers_iqr(input_df, target_column)
    stats = calculate_statistics(cleaned_df, target_column)
    return cleaned_df, stats
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Args:
        data (list or np.array): Input data array
        column (int): Index of column to process (if 2D array) or None for 1D array
        
    Returns:
        np.array: Data with outliers removed
    """
    if column is not None and len(data.shape) > 1:
        column_data = data[:, column]
    else:
        column_data = np.array(data)
    
    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    if column is not None and len(data.shape) > 1:
        mask = (data[:, column] >= lower_bound) & (data[:, column] <= upper_bound)
        return data[mask]
    else:
        mask = (column_data >= lower_bound) & (column_data <= upper_bound)
        return column_data[mask]

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Args:
        data (np.array): Input data array
        
    Returns:
        dict: Dictionary containing mean, median, std, min, max
    """
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data)
    }

def test_cleaner():
    """Test the outlier removal functionality."""
    test_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
    cleaned = remove_outliers_iqr(test_data, None)
    stats = calculate_statistics(cleaned)
    
    print(f"Original data: {test_data}")
    print(f"Cleaned data: {cleaned}")
    print(f"Statistics: {stats}")
    
    return cleaned, stats

if __name__ == "__main__":
    test_cleaner()