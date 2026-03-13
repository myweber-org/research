
import numpy as np

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data
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
    
    def detect_outliers(self, method='zscore', threshold=3):
        outlier_mask = pd.Series([False] * len(self.df), index=self.df.index)
        
        if method == 'zscore':
            for col in self.numerical_cols:
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                col_outliers = z_scores > threshold
                outlier_mask = outlier_mask | col_outliers.reindex(self.df.index, fill_value=False)
        
        elif method == 'iqr':
            for col in self.numerical_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                col_outliers = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                outlier_mask = outlier_mask | col_outliers
        
        return outlier_mask
    
    def remove_outliers(self, method='zscore', threshold=3):
        outlier_mask = self.detect_outliers(method, threshold)
        self.df = self.df[~outlier_mask].reset_index(drop=True)
        return self.df
    
    def get_clean_data(self):
        return self.df.copy()
    
    def summary(self):
        missing_counts = self.df.isnull().sum()
        outlier_info = self.detect_outliers()
        
        print("Data Cleaning Summary")
        print("=" * 50)
        print(f"Original shape: {self.df.shape}")
        print(f"Missing values per column:\n{missing_counts[missing_counts > 0]}")
        print(f"Outliers detected: {outlier_info.sum()} ({outlier_info.mean()*100:.2f}%)")
        print(f"Numerical columns: {len(self.numerical_cols)}")
        print(f"Categorical columns: {len(self.categorical_cols)}")

def example_usage():
    data = {
        'age': [25, 30, np.nan, 35, 150, 28, 32],
        'salary': [50000, 60000, 55000, np.nan, 1000000, 52000, 58000],
        'department': ['IT', 'HR', 'IT', 'Finance', 'IT', 'HR', np.nan]
    }
    
    df = pd.DataFrame(data)
    cleaner = DataCleaner(df)
    
    print("Original Data:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaner.handle_missing_values(strategy='mean')
    cleaner.remove_outliers(method='iqr')
    
    print("Cleaned Data:")
    print(cleaner.get_clean_data())
    print("\n" + "="*50 + "\n")
    
    cleaner.summary()

if __name__ == "__main__":
    example_usage()
def remove_duplicates_preserve_order(iterable):
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
        keep: {'first', 'last', False} which duplicates to keep
    
    Returns:
        DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to clean
    
    Returns:
        DataFrame with cleaned numeric columns
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of columns that must be present
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df is None:
        return False, "DataFrame is None"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"