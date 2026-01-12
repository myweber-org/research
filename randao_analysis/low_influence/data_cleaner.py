
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
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 200),
        'B': np.random.exponential(50, 200),
        'C': np.random.uniform(0, 1, 200)
    })
    sample_data.loc[5, 'A'] = 500
    sample_data.loc[10, 'B'] = 1000
    
    numeric_cols = ['A', 'B', 'C']
    result = clean_dataset(sample_data, numeric_cols)
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {result.shape}")
    print(f"Rows removed: {len(sample_data) - len(result)}")import pandas as pd
import numpy as np

def clean_csv_data(filepath, missing_strategy='mean', columns_to_drop=None):
    """
    Load and clean CSV data by handling missing values and removing specified columns.
    
    Parameters:
    filepath (str): Path to the CSV file
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'drop', 'zero')
    columns_to_drop (list): List of column names to remove from the dataset
    
    Returns:
    pandas.DataFrame: Cleaned dataframe
    """
    
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {filepath}")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {str(e)}")
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    if columns_to_drop:
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        print(f"Dropped {len(columns_to_drop)} specified columns")
    
    missing_before = df.isnull().sum().sum()
    
    if missing_before > 0:
        print(f"Found {missing_before} missing values")
        
        if missing_strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            print("Filled missing values with column means")
            
        elif missing_strategy == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            print("Filled missing values with column medians")
            
        elif missing_strategy == 'zero':
            df = df.fillna(0)
            print("Filled missing values with zeros")
            
        elif missing_strategy == 'drop':
            df = df.dropna()
            print("Dropped rows with missing values")
            
        else:
            raise ValueError(f"Unknown missing strategy: {missing_strategy}")
    
    missing_after = df.isnull().sum().sum()
    print(f"Missing values after cleaning: {missing_after}")
    print(f"Final data shape: {df.shape}")
    
    return df

def detect_outliers_iqr(df, column, threshold=1.5):
    """
    Detect outliers in a column using the Interquartile Range (IQR) method.
    
    Parameters:
    df (pandas.DataFrame): Input dataframe
    column (str): Column name to check for outliers
    threshold (float): IQR multiplier threshold
    
    Returns:
    pandas.Series: Boolean series indicating outliers
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    outlier_count = outliers.sum()
    if outlier_count > 0:
        print(f"Found {outlier_count} outliers in column '{column}'")
    
    return outliers

def save_cleaned_data(df, output_path):
    """
    Save cleaned dataframe to CSV file.
    
    Parameters:
    df (pandas.DataFrame): Dataframe to save
    output_path (str): Path for output CSV file
    """
    try:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    except Exception as e:
        raise Exception(f"Error saving data: {str(e)}")

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'value': [10.5, None, 15.2, 1000.0, 12.8],
        'category': ['A', 'B', None, 'A', 'C'],
        'score': [85, 92, 78, None, 88]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', missing_strategy='mean', columns_to_drop=['id'])
    
    outliers = detect_outliers_iqr(cleaned_df, 'value')
    
    save_cleaned_data(cleaned_df, 'cleaned_test_data.csv')