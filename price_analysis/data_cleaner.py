
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
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        removed_count = self.original_shape[0] - df_clean.shape[0]
        self.df = df_clean
        return removed_count
    
    def normalize_data(self, columns=None, method='minmax'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in df_normalized.columns:
                if method == 'minmax':
                    min_val = df_normalized[col].min()
                    max_val = df_normalized[col].max()
                    if max_val != min_val:
                        df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
                elif method == 'zscore':
                    mean_val = df_normalized[col].mean()
                    std_val = df_normalized[col].std()
                    if std_val > 0:
                        df_normalized[col] = (df_normalized[col] - mean_val) / std_val
        
        self.df = df_normalized
        return self.df
    
    def handle_missing_values(self, strategy='mean', fill_value=None):
        df_filled = self.df.copy()
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df_filled[col].isnull().any():
                if strategy == 'mean':
                    fill_val = df_filled[col].mean()
                elif strategy == 'median':
                    fill_val = df_filled[col].median()
                elif strategy == 'mode':
                    fill_val = df_filled[col].mode()[0]
                elif strategy == 'constant' and fill_value is not None:
                    fill_val = fill_value
                else:
                    continue
                    
                df_filled[col] = df_filled[col].fillna(fill_val)
        
        self.df = df_filled
        return self.df
    
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

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 50), 'feature_a'] = np.nan
    df.loc[np.random.choice(df.index, 30), 'feature_b'] = np.nan
    
    outliers = np.random.choice(df.index, 20)
    df.loc[outliers, 'feature_a'] = df['feature_a'].mean() + 5 * df['feature_a'].std()
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Initial data shape:", cleaner.original_shape)
    print("Missing values before:", sample_df.isnull().sum().sum())
    
    cleaner.handle_missing_values(strategy='mean')
    removed = cleaner.remove_outliers_iqr(factor=1.5)
    cleaner.normalize_data(method='minmax')
    
    cleaned_df = cleaner.get_cleaned_data()
    summary = cleaner.get_summary()
    
    print(f"Outliers removed: {removed}")
    print("Cleaned data shape:", cleaned_df.shape)
    print("Missing values after:", cleaned_df.isnull().sum().sum())
    print("Summary:", summary)import pandas as pd

def clean_dataset(df, columns_to_check=None):
    """
    Clean a pandas DataFrame by removing null values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): Specific columns to check for nulls.
            If None, checks all columns. Defaults to None.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove rows with null values
    if columns_to_check is None:
        cleaned_df = cleaned_df.dropna()
    else:
        cleaned_df = cleaned_df.dropna(subset=columns_to_check)
    
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_data(df, required_columns):
    """
    Validate that DataFrame contains all required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        bool: True if all required columns are present, False otherwise.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', None, 'David', 'Eve', 'Eve'],
        'age': [25, 30, 35, None, 28, 28],
        'score': [85.5, 92.0, 78.5, 88.0, 95.5, 95.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print(f"Shape: {df.shape}")
    
    # Clean the data
    cleaned_df = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print(f"Shape: {cleaned_df.shape}")
    
    # Validate required columns
    required_cols = ['id', 'name', 'age']
    is_valid = validate_data(cleaned_df, required_cols)
    print(f"\nData validation result: {is_valid}")