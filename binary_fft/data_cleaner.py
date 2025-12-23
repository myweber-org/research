
import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    columns_to_check (list, optional): List of column names to check for duplicates.
                                       If None, checks all columns.
    fill_missing (str or dict): Method to fill missing values.
                                Can be 'mean', 'median', 'mode', or a dictionary
                                of column:value pairs for custom filling.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicates
    if columns_to_check is None:
        columns_to_check = cleaned_df.columns.tolist()
    
    cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check, keep='first')
    
    # Handle missing values
    for column in cleaned_df.columns:
        if cleaned_df[column].isnull().any():
            if isinstance(fill_missing, dict):
                # Use custom value if provided in dictionary
                if column in fill_missing:
                    cleaned_df[column] = cleaned_df[column].fillna(fill_missing[column])
                else:
                    # If column not in dictionary, use column mean for numeric, mode for categorical
                    if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                        cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mean())
                    else:
                        cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0])
            elif fill_missing == 'mean':
                if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                    cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mean())
                else:
                    cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0])
            elif fill_missing == 'median':
                if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                    cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
                else:
                    cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0])
            elif fill_missing == 'mode':
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0])
            else:
                # Default to mean for numeric, mode for categorical
                if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                    cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mean())
                else:
                    cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0])
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate a DataFrame for basic integrity checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list, optional): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Dataset is valid"
import pandas as pd
import numpy as np
from datetime import datetime

def clean_dataframe(df):
    """
    Clean a pandas DataFrame by handling missing values,
    converting data types, and removing duplicates.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Fill missing numeric values with column median
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    
    # Fill missing categorical values with mode
    categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
    
    # Convert date columns to datetime
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == 'object':
            try:
                cleaned_df[col] = pd.to_datetime(cleaned_df[col])
            except (ValueError, TypeError):
                pass
    
    # Remove outliers using IQR method for numeric columns
    for col in numeric_cols:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    """
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    return True

def process_csv_file(input_path, output_path, required_columns=None):
    """
    Main function to process CSV file through cleaning pipeline.
    """
    try:
        # Read CSV file
        df = pd.read_csv(input_path)
        
        # Validate data
        validate_dataframe(df, required_columns)
        
        # Clean data
        cleaned_df = clean_dataframe(df)
        
        # Save cleaned data
        cleaned_df.to_csv(output_path, index=False)
        
        print(f"Data cleaning completed. Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
        print(f"Cleaned data saved to: {output_path}")
        
        return cleaned_df
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    # Define required columns if needed
    required_cols = ['id', 'name', 'value']
    
    try:
        result = process_csv_file(input_file, output_file, required_cols)
        print("Processing completed successfully")
    except Exception as e:
        print(f"Processing failed: {e}")import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
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
    
    return filtered_df

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def normalize_column(df, column, method='minmax'):
    """
    Normalize a DataFrame column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val != 0:
            df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

def handle_missing_values(df, column, strategy='mean'):
    """
    Handle missing values in a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    strategy (str): Strategy for handling missing values
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if strategy == 'mean':
        fill_value = df_copy[column].mean()
    elif strategy == 'median':
        fill_value = df_copy[column].median()
    elif strategy == 'mode':
        fill_value = df_copy[column].mode()[0]
    elif strategy == 'drop':
        df_copy = df_copy.dropna(subset=[column])
        return df_copy
    else:
        raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
    
    df_copy[column] = df_copy[column].fillna(fill_value)
    
    return df_copy

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    tuple: (is_valid, message)
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
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.numeric_columns
        
        clean_df = self.df.copy()
        
        for col in columns:
            if col not in self.numeric_columns:
                continue
                
            Q1 = clean_df[col].quantile(0.25)
            Q3 = clean_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            mask = (clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)
            clean_df = clean_df[mask]
        
        return clean_df
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.numeric_columns
        
        clean_df = self.df.copy()
        
        for col in columns:
            if col not in self.numeric_columns:
                continue
                
            z_scores = np.abs(stats.zscore(clean_df[col].dropna()))
            mask = z_scores < threshold
            clean_df = clean_df[mask]
        
        return clean_df
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        normalized_df = self.df.copy()
        
        for col in columns:
            if col not in self.numeric_columns:
                continue
                
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            
            if max_val > min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
        
        return normalized_df
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        normalized_df = self.df.copy()
        
        for col in columns:
            if col not in self.numeric_columns:
                continue
                
            mean_val = normalized_df[col].mean()
            std_val = normalized_df[col].std()
            
            if std_val > 0:
                normalized_df[col] = (normalized_df[col] - mean_val) / std_val
        
        return normalized_df
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        filled_df = self.df.copy()
        
        for col in columns:
            if col not in self.numeric_columns:
                continue
                
            mean_val = filled_df[col].mean()
            filled_df[col] = filled_df[col].fillna(mean_val)
        
        return filled_df
    
    def get_summary(self):
        summary = {
            'original_shape': self.df.shape,
            'numeric_columns': self.numeric_columns,
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.uniform(0, 1, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    data['feature_a'][np.random.choice(100, 5)] = np.nan
    data['feature_b'][np.random.choice(100, 3)] = 1000
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Original data shape:", cleaner.df.shape)
    print("\nMissing values:")
    print(cleaner.df.isnull().sum())
    
    cleaned_df = cleaner.remove_outliers_iqr(['feature_b'])
    print("\nAfter IQR outlier removal:", cleaned_df.shape)
    
    normalized_df = cleaner.normalize_minmax(['feature_a', 'feature_b', 'feature_c'])
    print("\nAfter min-max normalization:")
    print(normalized_df[['feature_a', 'feature_b', 'feature_c']].describe())
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    duplicates_removed = original_shape[0] - df_cleaned.shape[0]
    
    # Handle missing values
    if columns_to_check is None:
        columns_to_check = df_cleaned.columns
    
    missing_counts = df_cleaned[columns_to_check].isnull().sum()
    
    if fill_missing == 'mean':
        for col in columns_to_check:
            if df_cleaned[col].dtype in ['float64', 'int64']:
                df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
    elif fill_missing == 'median':
        for col in columns_to_check:
            if df_cleaned[col].dtype in ['float64', 'int64']:
                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
    elif fill_missing == 'mode':
        for col in columns_to_check:
            df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
    elif fill_missing == 'drop':
        df_cleaned = df_cleaned.dropna(subset=columns_to_check)
    
    # Calculate statistics
    final_missing = df_cleaned[columns_to_check].isnull().sum().sum()
    
    print(f"Original dataset shape: {original_shape}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Missing values before cleaning: {missing_counts.sum()}")
    print(f"Missing values after cleaning: {final_missing}")
    print(f"Cleaned dataset shape: {df_cleaned.shape}")
    
    return df_cleaned

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the structure and content of a DataFrame.
    """
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if len(df) < min_rows:
        raise ValueError(f"DataFrame must have at least {min_rows} rows")
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, np.nan, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'C', None, 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    try:
        validate_data(cleaned_df, required_columns=['id', 'value'], min_rows=3)
        print("\nData validation passed!")
    except ValueError as e:
        print(f"\nData validation failed: {e}")