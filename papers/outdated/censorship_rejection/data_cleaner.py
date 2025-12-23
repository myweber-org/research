
import pandas as pd
import numpy as np

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
    stats = {
        'original_count': len(df),
        'cleaned_count': len(remove_outliers_iqr(df, column)),
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std_dev': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max()
    }
    
    return stats

def process_dataset(file_path, column_to_clean):
    """
    Main function to load and clean a dataset.
    
    Parameters:
    file_path (str): Path to CSV file
    column_to_clean (str): Column name to clean
    
    Returns:
    tuple: (cleaned DataFrame, summary statistics)
    """
    try:
        df = pd.read_csv(file_path)
        cleaned_df = remove_outliers_iqr(df, column_to_clean)
        stats = calculate_summary_statistics(df, column_to_clean)
        
        return cleaned_df, stats
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None, None
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        return None, None

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'values': np.concatenate([
            np.random.normal(100, 15, 95),
            np.random.normal(300, 50, 5)
        ])
    })
    
    cleaned_data = remove_outliers_iqr(sample_data, 'values')
    print(f"Original data points: {len(sample_data)}")
    print(f"Cleaned data points: {len(cleaned_data)}")
    print(f"Outliers removed: {len(sample_data) - len(cleaned_data)}")