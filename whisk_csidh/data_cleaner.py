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

def process_dataset(file_path, column_name):
    """
    Load a dataset from file and clean it by removing outliers.
    
    Parameters:
    file_path (str): Path to the data file
    column_name (str): Column to clean
    
    Returns:
    tuple: (cleaned DataFrame, original stats, cleaned stats)
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    original_stats = calculate_summary_statistics(df, column_name)
    cleaned_df = remove_outliers_iqr(df, column_name)
    cleaned_stats = calculate_summary_statistics(cleaned_df, column_name)
    
    return cleaned_df, original_stats, cleaned_stats

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'values': np.concatenate([
            np.random.normal(100, 15, 90),
            np.random.normal(300, 50, 10)
        ])
    })
    
    print("Original data shape:", sample_data.shape)
    print("Original statistics:", calculate_summary_statistics(sample_data, 'values'))
    
    cleaned_data = remove_outliers_iqr(sample_data, 'values')
    print("\nCleaned data shape:", cleaned_data.shape)
    print("Cleaned statistics:", calculate_summary_statistics(cleaned_data, 'values'))import pandas as pd
import numpy as np
from scipy import stats

def load_data(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    """Remove outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    """Normalize column using min-max scaling."""
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(input_file, output_file):
    """Main function to clean dataset."""
    df = load_data(input_file)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
        df = normalize_column(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    
    return df

if __name__ == "__main__":
    cleaned_df = clean_dataset('raw_data.csv', 'cleaned_data.csv')
import pandas as pd
import numpy as np

def clean_csv_data(file_path, fill_method='mean'):
    """
    Load and clean CSV data by handling missing values.
    
    Args:
        file_path (str): Path to the CSV file.
        fill_method (str): Method for filling missing values ('mean', 'median', 'mode', 'zero').
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            print(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns
            
            for col in df.columns:
                if df[col].isnull().any():
                    if col in numeric_cols:
                        if fill_method == 'mean':
                            fill_value = df[col].mean()
                        elif fill_method == 'median':
                            fill_value = df[col].median()
                        elif fill_method == 'zero':
                            fill_value = 0
                        else:
                            fill_value = df[col].mean()
                        df[col].fillna(fill_value, inplace=True)
                        print(f"Filled missing values in '{col}' with {fill_method}: {fill_value}")
                    
                    elif col in categorical_cols:
                        mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                        df[col].fillna(mode_value, inplace=True)
                        print(f"Filled missing values in '{col}' with mode: {mode_value}")
        
        print(f"Cleaned data shape: {df.shape}")
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Args:
        df (pandas.DataFrame): Input DataFrame.
        column (str): Column name to process.
    
    Returns:
        pandas.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame")
        return df
    
    if not np.issubdtype(df[column].dtype, np.number):
        print(f"Column '{column}' is not numeric")
        return df
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    original_count = len(df)
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    removed_count = original_count - len(df_clean)
    
    print(f"Removed {removed_count} outliers from '{column}' (IQR method)")
    print(f"Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
    
    return df_clean

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV.
    
    Args:
        df (pandas.DataFrame): Cleaned DataFrame.
        output_path (str): Path to save the cleaned data.
    """
    if df is not None:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    else:
        print("No data to save")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, fill_method='median')
    
    if cleaned_df is not None:
        for col in cleaned_df.select_dtypes(include=[np.number]).columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        
        save_cleaned_data(cleaned_df, output_file)
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from the specified column.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_basic_stats(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std()
    }
    return stats