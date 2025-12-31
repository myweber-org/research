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

def clean_missing_values(data, strategy='mean'):
    """
    Handle missing values with different strategies
    """
    cleaned_data = data.copy()
    
    for column in cleaned_data.columns:
        if cleaned_data[column].isnull().any():
            if strategy == 'mean':
                fill_value = cleaned_data[column].mean()
            elif strategy == 'median':
                fill_value = cleaned_data[column].median()
            elif strategy == 'mode':
                fill_value = cleaned_data[column].mode()[0]
            elif strategy == 'ffill':
                cleaned_data[column] = cleaned_data[column].ffill()
                continue
            elif strategy == 'bfill':
                cleaned_data[column] = cleaned_data[column].bfill()
                continue
            else:
                fill_value = 0
            
            cleaned_data[column] = cleaned_data[column].fillna(fill_value)
    
    return cleaned_data

def validate_data_types(data, expected_types):
    """
    Validate that columns have expected data types
    """
    validation_results = {}
    
    for column, expected_type in expected_types.items():
        if column in data.columns:
            actual_type = str(data[column].dtype)
            validation_results[column] = {
                'expected': expected_type,
                'actual': actual_type,
                'valid': actual_type == expected_type
            }
    
    return validation_results

def create_data_summary(data):
    """
    Create a comprehensive summary of the dataset
    """
    summary = {
        'shape': data.shape,
        'columns': list(data.columns),
        'dtypes': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'unique_values': {col: data[col].nunique() for col in data.columns},
        'basic_stats': data.describe().to_dict()
    }
    
    return summary
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None):
    """
    Clean CSV data by handling missing values and standardizing columns.
    """
    try:
        df = pd.read_csv(input_path)
        
        # Remove completely empty rows and columns
        df.dropna(how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        
        # Fill numeric missing values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Remove duplicate rows
        df.drop_duplicates(inplace=True)
        
        # Reset index after cleaning
        df.reset_index(drop=True, inplace=True)
        
        # Save cleaned data
        if output_path is None:
            input_file = Path(input_path)
            output_path = input_file.parent / f"cleaned_{input_file.name}"
        
        df.to_csv(output_path, index=False)
        
        print(f"Data cleaning completed. Cleaned data saved to: {output_path}")
        print(f"Original shape: {df.shape}")
        print(f"Missing values after cleaning:\n{df.isnull().sum()}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return None
    except pd.errors.EmptyDataError:
        print("Error: Input file is empty")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df):
    """
    Validate dataframe for common data quality issues.
    """
    if df is None or df.empty:
        print("DataFrame is empty or None")
        return False
    
    checks = {
        'has_null_values': df.isnull().sum().sum() == 0,
        'has_duplicates': not df.duplicated().any(),
        'has_infinite_values': np.isfinite(df.select_dtypes(include=[np.number])).all().all(),
        'has_empty_strings': (df.select_dtypes(include=['object']) == '').sum().sum() == 0
    }
    
    print("Data Validation Results:")
    for check_name, check_result in checks.items():
        status = "PASS" if check_result else "FAIL"
        print(f"  {check_name}: {status}")
    
    return all(checks.values())

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Name': ['Alice', 'Bob', 'Charlie', None, 'Eve'],
        'Age': [25, None, 30, 35, 40],
        'Salary': [50000, 60000, None, 70000, 80000],
        'Department': ['IT', 'HR', 'IT', None, 'Finance']
    }
    
    # Create test dataframe
    test_df = pd.DataFrame(sample_data)
    test_csv = "test_data.csv"
    test_df.to_csv(test_csv, index=False)
    
    # Clean the data
    cleaned_df = clean_csv_data(test_csv)
    
    # Validate cleaned data
    if cleaned_df is not None:
        validate_dataframe(cleaned_df)
    
    # Clean up test file
    import os
    if os.path.exists(test_csv):
        os.remove(test_csv)
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

if __name__ == "__main__":
    sample_data = [3, 1, 2, 1, 4, 3, 5, 2]
    cleaned = remove_duplicates_preserve_order(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_clean = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                df_clean = df_clean[mask]
        
        removed_count = self.original_shape[0] - df_clean.shape[0]
        self.df = df_clean.reset_index(drop=True)
        return removed_count
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val != min_val:
                    df_normalized[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        self.df = df_normalized
        return self
    
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_standardized = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    df_standardized[col] = (self.df[col] - mean_val) / std_val
        
        self.df = df_standardized
        return self
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_filled = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                median_val = self.df[col].median()
                df_filled[col] = self.df[col].fillna(median_val)
        
        self.df = df_filled
        return self
    
    def get_cleaned_data(self):
        return self.df.copy()
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'missing_values': self.df.isnull().sum().sum()
        }
        return summary

def clean_dataset(df, remove_outliers=True, normalize=True, fill_missing=True):
    cleaner = DataCleaner(df)
    
    if fill_missing:
        cleaner.fill_missing_median()
    
    if remove_outliers:
        cleaner.remove_outliers_iqr()
    
    if normalize:
        cleaner.normalize_minmax()
    
    return cleaner.get_cleaned_data(), cleaner.get_summary()import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_na_method=None, fill_value=None):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default True.
    fill_na_method (str): Method to fill NA values ('ffill', 'bfill', or None). Default None.
    fill_value: Value to fill NA with if fill_na_method is None but fill_value is provided.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_na_method is not None:
        if fill_na_method == 'ffill':
            cleaned_df = cleaned_df.fillna(method='ffill')
        elif fill_na_method == 'bfill':
            cleaned_df = cleaned_df.fillna(method='bfill')
    elif fill_value is not None:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, None, 4, 2],
        'B': [5, None, 7, 8, 5],
        'C': [9, 10, 11, None, 9]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_value=0)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {is_valid}, Message: {message}")import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str): Method to fill missing values. 
                           Options: 'mean', 'median', 'mode', or 'drop'. 
                           Default is 'mean'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Removed {len(df) - len(cleaned_df)} duplicate rows.")
    
    if fill_missing == 'drop':
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.dropna()
        print(f"Dropped {initial_rows - len(cleaned_df)} rows with missing values.")
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[col].median()
                elif fill_missing == 'mode':
                    fill_value = cleaned_df[col].mode()[0]
                
                cleaned_df[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in column '{col}' with {fill_missing}: {fill_value:.2f}")
    
    print(f"Data cleaning complete. Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        dict: Dictionary with validation results.
    """
    validation_results = {
        'is_dataframe': isinstance(df, pd.DataFrame),
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_cols
        validation_results['all_required_columns_present'] = len(missing_cols) == 0
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None, 6],
        'B': [10, None, 30, 40, 50, 60],
        'C': ['x', 'y', 'y', 'z', None, 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation Results:")
    print(validate_dataframe(df))
    
    cleaned = clean_dataset(df, fill_missing='median')
    print("\nCleaned DataFrame:")
    print(cleaned)