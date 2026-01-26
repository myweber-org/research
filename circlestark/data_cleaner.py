import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (bool): Whether to fill missing values. Default is False.
    fill_value: Value to use for filling missing data. Default is 0.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate that the DataFrame meets certain criteria.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (bool, str) indicating validation result and message.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "Data validation passed"

def remove_outliers(df, column, threshold=3):
    """
    Remove outliers from a specific column using z-score method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to check for outliers.
    threshold (float): Z-score threshold for outlier detection.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    from scipy import stats
    import numpy as np
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    outlier_mask = z_scores < threshold
    filtered_df = df[outlier_mask].reset_index(drop=True)
    
    return filtered_dfimport numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val == min_val:
        return data[column]
    return (data[column] - min_val) / (max_val - min_val)

def standardize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column]
    return (data[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns, outlier_removal=True, normalization='standard'):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        if outlier_removal:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        
        if normalization == 'minmax':
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
        elif normalization == 'standard':
            cleaned_df[col] = standardize_zscore(cleaned_df, col)
    
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df, required_columns):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    return True
import pandas as pd
import numpy as np

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
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(filepath, numeric_columns):
    df = pd.read_csv(filepath)
    
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
            df = normalize_minmax(df, col)
    
    cleaned_filepath = filepath.replace('.csv', '_cleaned.csv')
    df.to_csv(cleaned_filepath, index=False)
    return cleaned_filepath

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_path = clean_dataset('sample_data.csv', ['feature1', 'feature2'])
    print(f"Cleaned data saved to: {cleaned_path}")
import pandas as pd
import numpy as np

def clean_csv_data(file_path, strategy='mean', fill_value=None):
    """
    Clean a CSV file by handling missing values.
    
    Args:
        file_path (str): Path to the CSV file.
        strategy (str): Strategy for handling missing values.
            Options: 'mean', 'median', 'mode', 'constant', 'drop'.
        fill_value: Value to use when strategy is 'constant'.
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        
        if strategy == 'drop':
            df_cleaned = df.dropna()
        elif strategy in ['mean', 'median', 'mode']:
            for column in df.select_dtypes(include=[np.number]).columns:
                if strategy == 'mean':
                    fill_val = df[column].mean()
                elif strategy == 'median':
                    fill_val = df[column].median()
                elif strategy == 'mode':
                    fill_val = df[column].mode()[0] if not df[column].mode().empty else 0
                
                df[column] = df[column].fillna(fill_val)
            df_cleaned = df
        elif strategy == 'constant':
            if fill_value is not None:
                df_cleaned = df.fillna(fill_value)
            else:
                raise ValueError("fill_value must be provided for constant strategy")
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return df_cleaned
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The file is empty")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to a CSV file.
    
    Args:
        df (pandas.DataFrame): DataFrame to save.
        output_path (str): Path for the output CSV file.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        if df is not None and not df.empty:
            df.to_csv(output_path, index=False)
            return True
        else:
            print("Error: DataFrame is empty or None")
            return False
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        return False

if __name__ == "__main__":
    input_file = "data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, strategy='mean')
    
    if cleaned_df is not None:
        success = save_cleaned_data(cleaned_df, output_file)
        if success:
            print(f"Data cleaned and saved to {output_file}")
        else:
            print("Failed to save cleaned data")
    else:
        print("Data cleaning failed")