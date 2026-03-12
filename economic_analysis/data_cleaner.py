
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_method='iqr'):
    """
    Clean a dataset by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    missing_strategy (str): Strategy for missing values ('mean', 'median', 'mode', 'drop')
    outlier_method (str): Method for outlier detection ('iqr', 'zscore')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif missing_strategy == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    elif missing_strategy == 'mode':
        cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    elif missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    
    # Handle outliers for numeric columns
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    if outlier_method == 'iqr':
        for col in numeric_cols:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers
            cleaned_df[col] = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
    
    elif outlier_method == 'zscore':
        for col in numeric_cols:
            z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
            mask = z_scores > 3
            cleaned_df.loc[mask, col] = cleaned_df[col].mean()
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    
    if len(df) < min_rows:
        return False, f"Dataset must have at least {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Dataset is valid"

def get_dataset_summary(df):
    """
    Generate a summary of the dataset.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    dict: Summary statistics
    """
    
    summary = {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
    }
    
    return summary

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'A': [1, 2, None, 4, 100],
        'B': [5, 6, 7, None, 9],
        'C': ['a', 'b', 'c', 'd', 'e']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nDataset summary:")
    print(get_dataset_summary(df))
    
    # Clean the dataset
    cleaned_df = clean_dataset(df, missing_strategy='mean', outlier_method='iqr')
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    # Validate the cleaned dataset
    is_valid, message = validate_data(cleaned_df, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {message}")
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_missing(self, threshold=0.3):
        missing_percent = self.df.isnull().sum() / len(self.df)
        columns_to_drop = missing_percent[missing_percent > threshold].index
        self.df = self.df.drop(columns=columns_to_drop)
        return self
    
    def fill_numeric_missing(self, method='median'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'median':
            for col in numeric_cols:
                self.df[col] = self.df[col].fillna(self.df[col].median())
        elif method == 'mean':
            for col in numeric_cols:
                self.df[col] = self.df[col].fillna(self.df[col].mean())
        elif method == 'mode':
            for col in numeric_cols:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                
        return self
    
    def detect_outliers_zscore(self, threshold=3):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers_mask = pd.Series([False] * len(self.df))
        
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(self.df[col].fillna(self.df[col].median())))
            outliers_mask = outliers_mask | (z_scores > threshold)
            
        return outliers_mask
    
    def remove_outliers(self, threshold=3):
        outliers_mask = self.detect_outliers_zscore(threshold)
        self.df = self.df[~outliers_mask].reset_index(drop=True)
        return self
    
    def normalize_numeric(self, method='minmax'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            for col in numeric_cols:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val != min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                    
        elif method == 'standard':
            for col in numeric_cols:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val != 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
                    
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_summary(self):
        cleaned_shape = self.df.shape
        rows_removed = self.original_shape[0] - cleaned_shape[0]
        cols_removed = self.original_shape[1] - cleaned_shape[1]
        
        summary = {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': cleaned_shape[0],
            'cleaned_columns': cleaned_shape[1],
            'rows_removed': rows_removed,
            'columns_removed': cols_removed,
            'missing_values': self.df.isnull().sum().sum()
        }
        
        return summary

def clean_dataset(df, missing_threshold=0.3, outlier_threshold=3, normalize=True):
    cleaner = DataCleaner(df)
    
    cleaner.remove_missing(missing_threshold)
    cleaner.fill_numeric_missing('median')
    cleaner.remove_outliers(outlier_threshold)
    
    if normalize:
        cleaner.normalize_numeric('standard')
    
    return cleaner.get_cleaned_data(), cleaner.get_summary()