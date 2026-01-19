import pandas as pd
import numpy as np

def clean_dataset(df, drop_threshold=0.5, fill_strategy='median'):
    """
    Clean a pandas DataFrame by handling missing values and standardizing columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_threshold (float): Threshold for dropping columns with too many nulls (0.0 to 1.0)
    fill_strategy (str): Strategy for filling missing values ('median', 'mean', 'mode', or 'constant')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Standardize column names
    df_clean.columns = df_clean.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Drop columns with too many null values
    null_ratio = df_clean.isnull().sum() / len(df_clean)
    columns_to_drop = null_ratio[null_ratio > drop_threshold].index
    df_clean = df_clean.drop(columns=columns_to_drop)
    
    # Fill missing values based on strategy
    for column in df_clean.columns:
        if df_clean[column].isnull().any():
            if fill_strategy == 'median' and pd.api.types.is_numeric_dtype(df_clean[column]):
                df_clean[column] = df_clean[column].fillna(df_clean[column].median())
            elif fill_strategy == 'mean' and pd.api.types.is_numeric_dtype(df_clean[column]):
                df_clean[column] = df_clean[column].fillna(df_clean[column].mean())
            elif fill_strategy == 'mode':
                df_clean[column] = df_clean[column].fillna(df_clean[column].mode()[0])
            elif fill_strategy == 'constant':
                if pd.api.types.is_numeric_dtype(df_clean[column]):
                    df_clean[column] = df_clean[column].fillna(0)
                else:
                    df_clean[column] = df_clean[column].fillna('unknown')
    
    # Remove duplicate rows
    df_clean = df_clean.drop_duplicates()
    
    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for common data quality issues.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of columns that must be present
    
    Returns:
    dict: Dictionary containing validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'null_values': int(df.isnull().sum().sum()),
        'duplicate_rows': df.duplicated().sum(),
        'column_types': {col: str(df[col].dtype) for col in df.columns}
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_columns
    
    return validation_results

# Example usage (commented out for production)
if __name__ == "__main__":
    # Create sample data for testing
    sample_data = {
        'Customer ID': [1, 2, 3, 4, 5],
        'First Name': ['Alice', 'Bob', None, 'David', 'Eve'],
        'Last Name': ['Smith', 'Johnson', 'Williams', None, 'Brown'],
        'Age': [25, 30, 35, None, 28],
        'Salary': [50000, 60000, None, 75000, 55000],
        'Department': ['Sales', 'IT', 'IT', None, 'HR']
    }
    
    df_sample = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df_sample)
    print("\nValidation Results:")
    print(validate_dataset(df_sample))
    
    df_cleaned = clean_dataset(df_sample, drop_threshold=0.3, fill_strategy='median')
    print("\nCleaned DataFrame:")
    print(df_cleaned)
    print("\nCleaned Validation Results:")
    print(validate_dataset(df_cleaned))
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Args:
        df: pandas DataFrame
        column: Column name to process
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    """
    Normalize column values to range [0, 1] using min-max scaling.
    
    Args:
        df: pandas DataFrame
        column: Column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = df[column].min()
    max_val = df[column].max()
    
    if max_val == min_val:
        df[column + '_normalized'] = 0.5
    else:
        df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    
    return df

def clean_dataset(df, numeric_columns=None):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    
    Args:
        df: Input DataFrame
        numeric_columns: List of numeric columns to process
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    
    return cleaned_df

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column.
    
    Args:
        df: pandas DataFrame
        column: Column name
    
    Returns:
        Dictionary with statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 21),
        'value': [10, 12, 15, 18, 20, 22, 25, 28, 30, 32,
                  35, 38, 40, 42, 45, 48, 50, 100, 120, 150]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data:")
    print(df)
    
    cleaned = clean_dataset(df, ['value'])
    print("\nCleaned data:")
    print(cleaned)
    
    stats = calculate_statistics(df, 'value')
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
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
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                df_clean = df_clean[mask]
                
        self.df = df_clean
        removed_count = self.original_shape[0] - self.df.shape[0]
        print(f"Removed {removed_count} outliers")
        return self
        
    def normalize_data(self, columns=None, method='minmax'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        
        for col in columns:
            if col in self.df.columns:
                if method == 'minmax':
                    min_val = self.df[col].min()
                    max_val = self.df[col].max()
                    if max_val != min_val:
                        df_normalized[col] = (self.df[col] - min_val) / (max_val - min_val)
                    else:
                        df_normalized[col] = 0
                        
                elif method == 'zscore':
                    mean_val = self.df[col].mean()
                    std_val = self.df[col].std()
                    if std_val > 0:
                        df_normalized[col] = (self.df[col] - mean_val) / std_val
                    else:
                        df_normalized[col] = 0
                        
                elif method == 'robust':
                    median_val = self.df[col].median()
                    iqr_val = stats.iqr(self.df[col])
                    if iqr_val > 0:
                        df_normalized[col] = (self.df[col] - median_val) / iqr_val
                    else:
                        df_normalized[col] = 0
        
        self.df = df_normalized
        print(f"Normalized {len(columns)} columns using {method} method")
        return self
        
    def handle_missing_values(self, columns=None, strategy='mean'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_filled = self.df.copy()
        
        for col in columns:
            if col in self.df.columns and self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                elif strategy == 'drop':
                    df_filled = df_filled.dropna(subset=[col])
                    continue
                else:
                    fill_value = strategy
                    
                df_filled[col] = self.df[col].fillna(fill_value)
                print(f"Filled missing values in {col} with {fill_value}")
        
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

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[np.random.choice(df.index, 50), 'feature1'] = np.random.normal(300, 50, 50)
    df.loc[np.random.choice(df.index, 30), 'feature2'] = np.random.exponential(200, 30)
    df.loc[np.random.choice(df.index, 20), 'feature3'] = np.nan
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Original data shape:", sample_df.shape)
    
    cleaner = DataCleaner(sample_df)
    
    cleaned_df = (cleaner
                 .remove_outliers_iqr(['feature1', 'feature2'])
                 .handle_missing_values(strategy='median')
                 .normalize_data(method='minmax')
                 .get_cleaned_data())
    
    print("\nCleaned data shape:", cleaned_df.shape)
    print("\nSummary:")
    for key, value in cleaner.get_summary().items():
        print(f"{key}: {value}")