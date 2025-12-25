
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
    
    return filtered_df.copy()

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
    
    cleaned_df = remove_outliers_iqr(df, column)
    
    stats = {
        'original_count': len(df),
        'cleaned_count': len(cleaned_df),
        'outliers_removed': len(df) - len(cleaned_df),
        'original_mean': df[column].mean(),
        'cleaned_mean': cleaned_df[column].mean(),
        'original_std': df[column].std(),
        'cleaned_std': cleaned_df[column].std(),
        'original_median': df[column].median(),
        'cleaned_median': cleaned_df[column].median()
    }
    
    return stats

def process_dataframe(df, columns_to_clean=None):
    """
    Process multiple columns in a DataFrame for outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean. If None, uses all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of statistics for each processed column
    """
    if columns_to_clean is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns_to_clean = numeric_cols
    
    cleaned_df = df.copy()
    all_stats = {}
    
    for column in columns_to_clean:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
                stats = calculate_summary_statistics(df, column)
                all_stats[column] = stats
            except Exception as e:
                print(f"Error processing column '{column}': {e}")
    
    return cleaned_df, all_stats

def save_cleaned_data(df, output_path, format='csv'):
    """
    Save cleaned DataFrame to file.
    
    Parameters:
    df (pd.DataFrame): DataFrame to save
    output_path (str): Path to save the file
    format (str): File format ('csv' or 'parquet')
    
    Returns:
    bool: True if successful, False otherwise
    """
    try:
        if format.lower() == 'csv':
            df.to_csv(output_path, index=False)
        elif format.lower() == 'parquet':
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError("Format must be 'csv' or 'parquet'")
        
        print(f"Data successfully saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error saving data: {e}")
        return False