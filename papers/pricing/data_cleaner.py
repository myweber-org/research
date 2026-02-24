import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean':
            for col in self.numerical_cols:
                self.df[col].fillna(self.df[col].mean(), inplace=True)
        elif strategy == 'median':
            for col in self.numerical_cols:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        elif strategy == 'mode':
            for col in self.numerical_cols:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        elif strategy == 'constant':
            if fill_value is not None:
                self.df.fillna(fill_value, inplace=True)
        
        for col in self.categorical_cols:
            self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        
        return self.df
    
    def remove_outliers(self, method='zscore', threshold=3):
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(self.df[self.numerical_cols]))
            mask = (z_scores < threshold).all(axis=1)
            self.df = self.df[mask]
        elif method == 'iqr':
            for col in self.numerical_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        
        return self.df
    
    def get_cleaned_data(self):
        return self.df.copy()
    
    def summary(self):
        missing = self.df.isnull().sum()
        outliers = {}
        
        for col in self.numerical_cols:
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            outliers[col] = (z_scores > 3).sum()
        
        return {
            'missing_values': missing.to_dict(),
            'outliers_detected': outliers,
            'shape': self.df.shape
        }

def example_usage():
    data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, 20, 30, np.nan, 50, 60],
        'C': ['a', 'b', 'a', 'b', np.nan, 'a']
    }
    
    df = pd.DataFrame(data)
    cleaner = DataCleaner(df)
    
    print("Original data:")
    print(df)
    print("\nMissing values summary:")
    print(df.isnull().sum())
    
    cleaned_df = cleaner.handle_missing_values(strategy='mean')
    cleaned_df = cleaner.remove_outliers(method='zscore', threshold=3)
    
    print("\nCleaned data:")
    print(cleaned_df)
    print("\nCleaning summary:")
    print(cleaner.summary())
    
    return cleaned_df

if __name__ == "__main__":
    example_usage()import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicate rows and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    columns_to_check (list, optional): List of column names to check for duplicates.
                                       If None, checks all columns.
    fill_missing (str or value): Method to fill missing values.
                                 Options: 'mean', 'median', 'mode', or a scalar value.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    if columns_to_check is None:
        cleaned_df = cleaned_df.drop_duplicates()
    else:
        cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
    
    # Handle missing values
    if fill_missing == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif fill_missing == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                mode_val = cleaned_df[col].mode()
                if not mode_val.empty:
                    cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
    else:
        # Assume fill_missing is a scalar value
        cleaned_df = cleaned_df.fillna(fill_missing)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list, optional): List of column names that must be present.
    
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

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'A': [1, 2, 2, 4, None],
#         'B': [5, None, 7, 8, 9],
#         'C': ['x', 'y', 'y', 'z', None]
#     }
#     
#     df = pd.DataFrame(data)
#     print("Original DataFrame:")
#     print(df)
#     
#     # Clean the data
#     cleaned = clean_dataset(df, fill_missing='mean')
#     print("\nCleaned DataFrame:")
#     print(cleaned)
#     
#     # Validate
#     is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B'])
#     print(f"\nValidation: {is_valid}, Message: {message}")