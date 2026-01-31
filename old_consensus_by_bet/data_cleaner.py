
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def detect_outliers_iqr(self, column):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        return outliers
    
    def remove_outliers(self, columns=None, method='iqr'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        original_len = len(self.df)
        
        for col in columns:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                self.df = self.df[z_scores < 3]
        
        removed_count = original_len - len(self.df)
        print(f"Removed {removed_count} outliers")
        return self
    
    def normalize_column(self, column, method='minmax'):
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            self.df[column] = (self.df[column] - mean_val) / std_val
        return self
    
    def fill_missing(self, strategy='mean', custom_value=None):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                elif strategy == 'custom' and custom_value is not None:
                    fill_value = custom_value
                else:
                    continue
                
                self.df[col].fillna(fill_value, inplace=True)
        
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_summary(self):
        summary = {
            'original_shape': self.original_shape,
            'cleaned_shape': self.df.shape,
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns)
        }
        return summary

def example_usage():
    np.random.seed(42)
    data = {
        'age': np.random.normal(35, 10, 100),
        'income': np.random.normal(50000, 15000, 100),
        'score': np.random.uniform(0, 100, 100)
    }
    
    df = pd.DataFrame(data)
    df.loc[10:15, 'age'] = np.nan
    df.loc[5, 'income'] = 200000
    df.loc[6, 'income'] = -50000
    
    cleaner = DataCleaner(df)
    cleaner.fill_missing(strategy='mean')
    cleaner.remove_outliers(method='iqr')
    cleaner.normalize_column('income', method='minmax')
    
    cleaned_df = cleaner.get_cleaned_data()
    summary = cleaner.get_summary()
    
    return cleaned_df, summary

if __name__ == "__main__":
    cleaned_data, stats_summary = example_usage()
    print("Data cleaning completed")
    print(f"Original shape: {stats_summary['original_shape']}")
    print(f"Cleaned shape: {stats_summary['cleaned_shape']}")import pandas as pd

def clean_dataset(df):
    """
    Clean a pandas DataFrame by removing null values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to be cleaned.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Remove rows with any null values
    df_cleaned = df.dropna()
    
    # Remove duplicate rows
    df_cleaned = df_cleaned.drop_duplicates()
    
    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def validate_data(df, required_columns):
    """
    Validate that the DataFrame contains all required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        bool: True if all required columns are present, False otherwise.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    
    return True

def process_data_file(file_path, required_columns=None):
    """
    Load and clean data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        required_columns (list, optional): List of required columns for validation.
    
    Returns:
        pd.DataFrame or None: Cleaned DataFrame if successful, None otherwise.
    """
    try:
        # Load data from CSV
        df = pd.read_csv(file_path)
        
        # Validate required columns if provided
        if required_columns and not validate_data(df, required_columns):
            return None
        
        # Clean the data
        df_cleaned = clean_dataset(df)
        
        # Print cleaning statistics
        print(f"Original rows: {len(df)}")
        print(f"Cleaned rows: {len(df_cleaned)}")
        print(f"Rows removed: {len(df) - len(df_cleaned)}")
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The file is empty")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def clean_dataset(df, columns_to_clean=None):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns_to_clean is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        columns_to_clean = numeric_columns
    
    cleaned_df = df.copy()
    
    for column in columns_to_clean:
        if column in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[column]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[::100, 'A'] = 500
    
    print("Original dataset shape:", df.shape)
    print("Original statistics for column 'A':")
    print(calculate_summary_statistics(df, 'A'))
    
    cleaned_df = clean_dataset(df, ['A'])
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("Cleaned statistics for column 'A':")
    print(calculate_summary_statistics(cleaned_df, 'A'))
import pandas as pd
import numpy as np

def clean_dataframe(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        missing_strategy (str): Strategy for handling missing values. 
                               Options: 'mean', 'median', 'drop', 'fill_zero'
        outlier_threshold (float): Number of standard deviations for outlier detection
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean())
    elif missing_strategy == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median())
    elif missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif missing_strategy == 'fill_zero':
        cleaned_df = cleaned_df.fillna(0)
    else:
        raise ValueError(f"Unknown missing strategy: {missing_strategy}")
    
    # Handle outliers using z-score method
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if cleaned_df[col].notna().any():
            z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
            outlier_mask = z_scores > outlier_threshold
            
            if outlier_mask.any():
                # Cap outliers at threshold * std
                upper_bound = cleaned_df[col].mean() + outlier_threshold * cleaned_df[col].std()
                lower_bound = cleaned_df[col].mean() - outlier_threshold * cleaned_df[col].std()
                
                cleaned_df.loc[outlier_mask, col] = np.where(
                    cleaned_df.loc[outlier_mask, col] > upper_bound,
                    upper_bound,
                    np.where(
                        cleaned_df.loc[outlier_mask, col] < lower_bound,
                        lower_bound,
                        cleaned_df.loc[outlier_mask, col]
                    )
                )
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

# Example usage
if __name__ == "__main__":
    # Create sample data with missing values and outliers
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],  # Contains outlier (100) and missing value
        'B': [5, 6, 7, np.nan, 9],
        'C': [10, 11, 12, 13, 14]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nDataFrame info:")
    print(df.info())
    
    # Clean the data
    cleaned = clean_dataframe(df, missing_strategy='mean', outlier_threshold=2)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    # Validate the cleaned data
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {is_valid} - {message}")