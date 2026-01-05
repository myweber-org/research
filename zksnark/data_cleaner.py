
import pandas as pd

def clean_dataset(df, fill_method='mean'):
    """
    Remove duplicate rows and fill missing values in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        fill_method (str): Method for filling missing values ('mean', 'median', 'mode', or 'zero').
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Fill missing values
    for column in cleaned_df.select_dtypes(include=['number']).columns:
        if cleaned_df[column].isnull().any():
            if fill_method == 'mean':
                cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
            elif fill_method == 'median':
                cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
            elif fill_method == 'mode':
                cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
            elif fill_method == 'zero':
                cleaned_df[column].fillna(0, inplace=True)
    
    return cleaned_df

def validate_dataset(df):
    """
    Validate dataset by checking for remaining missing values and duplicates.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
    
    Returns:
        dict: Dictionary containing validation results.
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 10, 40, 50],
        'C': [100, 200, 300, 400, 500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_method='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    validation = validate_dataset(cleaned)
    print("\nValidation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    if max_val - min_val == 0:
        return df[column]
    return (df[column] - min_val) / (max_val - min_val)

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.dropna()

def calculate_statistics(df):
    stats = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        stats[col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'median': df[col].median()
        }
    return pd.DataFrame(stats).T

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 200),
        'B': np.random.exponential(2, 200),
        'C': np.random.uniform(0, 1, 200)
    })
    
    cleaned_data = clean_dataset(sample_data, ['A', 'B', 'C'])
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned_data.shape}")
    
    stats_df = calculate_statistics(cleaned_data)
    print("\nCleaned data statistics:")
    print(stats_df)