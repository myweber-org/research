import pandas as pd
import numpy as np

def remove_outliers_iqr(df, columns=None):
    """
    Remove outliers from specified columns using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to process. If None, process all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize specified columns in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_norm = df.copy()
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val != min_val:
                    df_norm[col] = (df[col] - min_val) / (max_val - min_val)
            
            elif method == 'zscore':
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val != 0:
                    df_norm[col] = (df[col] - mean_val) / std_val
    
    return df_norm

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    columns (list): List of column names to process
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    if columns is None:
        columns = df.columns.tolist()
    
    df_processed = df.copy()
    
    for col in columns:
        if col in df.columns and df[col].isnull().any():
            if strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                df_processed[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                df_processed[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mode':
                df_processed[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None, inplace=True)
            elif strategy == 'drop':
                df_processed = df_processed.dropna(subset=[col])
    
    return df_processed.reset_index(drop=True)
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None):
    """
    Clean CSV data by handling missing values and converting data types.
    """
    try:
        df = pd.read_csv(input_path)
        
        original_shape = df.shape
        print(f"Original data shape: {original_shape}")
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")
        
        # Handle missing values
        missing_before = df.isnull().sum().sum()
        
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_val)
        
        missing_after = df.isnull().sum().sum()
        print(f"Handled {missing_before - missing_after} missing values")
        
        # Convert data types
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                    print(f"Converted column '{col}' to datetime")
                except (ValueError, TypeError):
                    pass
        
        # Generate output path if not provided
        if output_path is None:
            input_file = Path(input_path)
            output_path = input_file.parent / f"cleaned_{input_file.name}"
        
        # Save cleaned data
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        print(f"Final data shape: {df.shape}")
        
        return df, output_path
        
    except FileNotFoundError:
        print(f"Error: File '{input_path}' not found")
        return None, None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_path}' is empty")
        return None, None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None, None

def validate_dataframe(df):
    """
    Validate dataframe for common data quality issues.
    """
    if df is None:
        return False
    
    validation_results = {
        'has_data': not df.empty,
        'row_count': len(df),
        'column_count': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    print("Data Validation Results:")
    for key, value in validation_results.items():
        print(f"  {key}: {value}")
    
    return validation_results['has_data'] and validation_results['missing_values'] == 0

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-03'],
        'temperature': [22.5, np.nan, 24.0, 22.5],
        'humidity': [45, 50, np.nan, 45],
        'location': ['NYC', 'NYC', 'LA', 'NYC'],
        'notes': ['Sunny', 'Cloudy', None, 'Sunny']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df, output_file = clean_csv_data('test_data.csv')
    
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df)
        print(f"Data validation passed: {is_valid}")
        
        # Clean up test file
        import os
        if os.path.exists('test_data.csv'):
            os.remove('test_data.csv')
        if os.path.exists(output_file):
            os.remove(output_file)