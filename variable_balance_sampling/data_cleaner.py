import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': len(df[column])
    }
    
    return stats

def process_numerical_data(df, columns):
    """
    Process multiple numerical columns by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to process
    
    Returns:
    pd.DataFrame: Processed DataFrame
    """
    processed_df = df.copy()
    
    for col in columns:
        if col in processed_df.columns and pd.api.types.is_numeric_dtype(processed_df[col]):
            processed_df = remove_outliers_iqr(processed_df, col)
    
    return processed_df

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 27, 28, 29, 100, -10],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 90, 10],
        'pressure': [1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1100, 900]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned_df = process_numerical_data(df, ['temperature', 'humidity', 'pressure'])
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n")
    
    for col in ['temperature', 'humidity', 'pressure']:
        stats = calculate_summary_statistics(cleaned_df, col)
        print(f"Statistics for {col}: {stats}")import pandas as pd
import numpy as np
from scipy import stats

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def standardize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def handle_missing_mean(data, column):
    mean_val = data[column].mean()
    data[column].fillna(mean_val, inplace=True)
    return data

def clean_dataset(dataframe, numeric_columns):
    df_clean = dataframe.copy()
    
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean = handle_missing_mean(df_clean, col)
            df_clean = remove_outliers(df_clean, col)
            df_clean = normalize_minmax(df_clean, col)
    
    return df_clean

def get_dataset_summary(dataframe):
    summary = {
        'original_shape': dataframe.shape,
        'numeric_columns': dataframe.select_dtypes(include=[np.number]).columns.tolist(),
        'missing_values': dataframe.isnull().sum().to_dict(),
        'data_types': dataframe.dtypes.to_dict()
    }
    return summary

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 100),
        'B': np.random.exponential(50, 100),
        'C': np.random.randint(1, 1000, 100)
    })
    
    sample_data.loc[10:15, 'A'] = np.nan
    sample_data.loc[20:25, 'B'] = np.nan
    
    print("Original dataset shape:", sample_data.shape)
    print("\nDataset summary:")
    summary = get_dataset_summary(sample_data)
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    numeric_cols = ['A', 'B', 'C']
    cleaned_data = clean_dataset(sample_data, numeric_cols)
    
    print("\nCleaned dataset shape:", cleaned_data.shape)
    print("\nCleaned data preview:")
    print(cleaned_data.head())