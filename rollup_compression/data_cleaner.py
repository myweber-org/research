import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file, missing_strategy='mean'):
    """
    Load a CSV file, clean missing values, and save the cleaned data.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the cleaned CSV file.
        missing_strategy (str): Strategy for handling missing values.
            Options: 'mean', 'median', 'drop', 'zero'.
    """
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded data from {input_file}. Shape: {df.shape}")
        
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values in the dataset.")
            
            if missing_strategy == 'mean':
                # Fill numeric columns with mean
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                print("Filled missing values with column means.")
                
            elif missing_strategy == 'median':
                # Fill numeric columns with median
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                print("Filled missing values with column medians.")
                
            elif missing_strategy == 'drop':
                # Drop rows with any missing values
                df = df.dropna()
                print(f"Dropped rows with missing values. New shape: {df.shape}")
                
            elif missing_strategy == 'zero':
                # Fill all missing values with 0
                df = df.fillna(0)
                print("Filled missing values with zeros.")
                
            else:
                print(f"Unknown strategy: {missing_strategy}. Using 'mean' as default.")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        else:
            print("No missing values found in the dataset.")
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}. Shape: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_data(df, required_columns=None):
    """
    Validate the cleaned dataframe for basic data quality.
    
    Args:
        df (pd.DataFrame): Dataframe to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        dict: Dictionary with validation results.
    """
    validation_results = {
        'is_valid': True,
        'issues': []
    }
    
    if df is None or df.empty:
        validation_results['is_valid'] = False
        validation_results['issues'].append('DataFrame is empty or None')
        return validation_results
    
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f'Missing required columns: {missing_cols}')
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    if inf_count > 0:
        validation_results['issues'].append(f'Found {inf_count} infinite values')
    
    # Check data types consistency
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check for mixed types in object columns
            unique_types = set([type(x).__name__ for x in df[col].dropna()])
            if len(unique_types) > 1:
                validation_results['issues'].append(f'Column {col} has mixed types: {unique_types}')
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    # Clean the data using mean imputation
    cleaned_df = clean_csv_data(input_csv, output_csv, missing_strategy='mean')
    
    if cleaned_df is not None:
        # Validate the cleaned data
        validation = validate_data(cleaned_df)
        
        if validation['is_valid']:
            print("Data validation passed.")
        else:
            print("Data validation failed. Issues:")
            for issue in validation['issues']:
                print(f"  - {issue}")
        
        # Show basic statistics
        print("\nBasic statistics:")
        print(cleaned_df.describe())
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    drop_duplicates (bool): Whether to remove duplicate rows
    fill_missing (bool): Whether to fill missing values
    strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'zero')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed_rows = initial_rows - len(cleaned_df)
        print(f"Removed {removed_rows} duplicate rows")
    
    if fill_missing and cleaned_df.isnull().sum().any():
        for column in cleaned_df.columns:
            if cleaned_df[column].dtype in ['int64', 'float64']:
                if strategy == 'mean':
                    fill_value = cleaned_df[column].mean()
                elif strategy == 'median':
                    fill_value = cleaned_df[column].median()
                elif strategy == 'mode':
                    fill_value = cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else 0
                elif strategy == 'zero':
                    fill_value = 0
                else:
                    fill_value = cleaned_df[column].mean()
                
                missing_count = cleaned_df[column].isnull().sum()
                if missing_count > 0:
                    cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                    print(f"Filled {missing_count} missing values in column '{column}' with {strategy} value: {fill_value}")
    
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    
    if len(df) < min_rows:
        print(f"Dataset has only {len(df)} rows, minimum required is {min_rows}")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

def main():
    sample_data = {
        'id': [1, 2, 3, 3, 4, 5],
        'value': [10.5, 20.3, np.nan, 30.1, 40.7, np.nan],
        'category': ['A', 'B', 'A', 'A', 'C', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nDataset info:")
    print(df.info())
    
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing=True, strategy='mean')
    
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    is_valid = validate_dataset(cleaned_df, required_columns=['id', 'value', 'category'], min_rows=3)
    print(f"\nDataset validation: {'PASS' if is_valid else 'FAIL'}")

if __name__ == "__main__":
    main()import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = self.df.select_dtypes(include=['object']).columns

    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean':
            for col in self.numeric_columns:
                self.df[col].fillna(self.df[col].mean(), inplace=True)
        elif strategy == 'median':
            for col in self.numeric_columns:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        elif strategy == 'mode':
            for col in self.numeric_columns:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        elif strategy == 'constant':
            if fill_value is not None:
                self.df.fillna(fill_value, inplace=True)
            else:
                raise ValueError("fill_value must be provided for constant strategy")
        else:
            raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', or 'constant'")
        
        for col in self.categorical_columns:
            self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        
        return self.df

    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.numeric_columns
        
        df_clean = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        self.df = df_clean
        return self.df

    def standardize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        df_standardized = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                mean = df_standardized[col].mean()
                std = df_standardized[col].std()
                if std > 0:
                    df_standardized[col] = (df_standardized[col] - mean) / std
        
        self.df = df_standardized
        return self.df

    def normalize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        df_normalized = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                min_val = df_normalized[col].min()
                max_val = df_normalized[col].max()
                if max_val > min_val:
                    df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        
        self.df = df_normalized
        return self.df

    def get_cleaned_data(self):
        return self.df.copy()

def example_usage():
    data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, 20, 30, np.nan, 50, 60],
        'C': ['a', 'b', 'a', np.nan, 'b', 'a'],
        'D': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaner = DataCleaner(df)
    
    cleaned_df = cleaner.handle_missing_values(strategy='mean')
    print("After handling missing values:")
    print(cleaned_df)
    print("\n")
    
    cleaned_df = cleaner.remove_outliers_iqr(columns=['A'], multiplier=1.5)
    print("After removing outliers from column A:")
    print(cleaned_df)
    print("\n")
    
    standardized_df = cleaner.standardize_data(columns=['D'])
    print("After standardizing column D:")
    print(standardized_df)
    print("\n")
    
    normalized_df = cleaner.normalize_data(columns=['D'])
    print("After normalizing column D:")
    print(normalized_df)

if __name__ == "__main__":
    example_usage()
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_clean = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
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
                if max_val > min_val:
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
        return self.df
    
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

def clean_dataset(df, outlier_columns=None, normalize=True, standardize=False, fill_missing=True):
    cleaner = DataCleaner(df)
    
    if outlier_columns:
        removed = cleaner.remove_outliers_iqr(outlier_columns)
        print(f"Removed {removed} outliers using IQR method")
    
    if fill_missing:
        cleaner.fill_missing_median()
        print("Filled missing values with median")
    
    if normalize:
        cleaner.normalize_minmax()
        print("Applied min-max normalization")
    
    if standardize:
        cleaner.standardize_zscore()
        print("Applied z-score standardization")
    
    summary = cleaner.get_summary()
    print(f"Cleaning complete. Removed {summary['rows_removed']} rows.")
    
    return cleaner.get_cleaned_data()import pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    """
    Remove rows with missing values from specified columns or entire DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of columns to check for missing values.
                                  If None, checks all columns.
    
    Returns:
        pd.DataFrame: DataFrame with rows containing missing values removed
    """
    if columns is None:
        return df.dropna()
    else:
        return df.dropna(subset=columns)

def fill_missing_with_mean(df, columns):
    """
    Fill missing values in specified columns with their mean values.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to fill missing values
    
    Returns:
        pd.DataFrame: DataFrame with missing values filled
    """
    df_filled = df.copy()
    for col in columns:
        if col in df.columns:
            df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
    return df_filled

def detect_outliers_iqr(df, column, threshold=1.5):
    """
    Detect outliers using the Interquartile Range (IQR) method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to check for outliers
        threshold (float): IQR multiplier for outlier detection
    
    Returns:
        pd.Series: Boolean series indicating outliers (True = outlier)
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def remove_outliers(df, column, threshold=1.5):
    """
    Remove outliers from a specific column using IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to remove outliers from
        threshold (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    outliers = detect_outliers_iqr(df, column, threshold)
    return df[~outliers].reset_index(drop=True)

def standardize_column(df, column):
    """
    Standardize a column to have zero mean and unit variance.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to standardize
    
    Returns:
        pd.DataFrame: DataFrame with standardized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_standardized = df.copy()
    mean_val = df_standardized[column].mean()
    std_val = df_standardized[column].std()
    
    if std_val > 0:
        df_standardized[column] = (df_standardized[column] - mean_val) / std_val
    
    return df_standardized

def clean_dataset(df, missing_strategy='remove', outlier_columns=None, standardize_columns=None):
    """
    Comprehensive data cleaning function.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        missing_strategy (str): Strategy for handling missing values ('remove' or 'mean')
        outlier_columns (list, optional): Columns to remove outliers from
        standardize_columns (list, optional): Columns to standardize
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'remove':
        cleaned_df = remove_missing_rows(cleaned_df)
    elif missing_strategy == 'mean':
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
        cleaned_df = fill_missing_with_mean(cleaned_df, numeric_cols)
    
    # Remove outliers
    if outlier_columns:
        for col in outlier_columns:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers(cleaned_df, col)
    
    # Standardize columns
    if standardize_columns:
        for col in standardize_columns:
            if col in cleaned_df.columns:
                cleaned_df = standardize_column(cleaned_df, col)
    
    return cleaned_df