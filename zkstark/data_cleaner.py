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
    
    def normalize_column(self, column, method='zscore'):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise ValueError(f"Column '{column}' is not numeric")
        
        if method == 'zscore':
            self.df[f'{column}_normalized'] = stats.zscore(self.df[column])
        elif method == 'minmax':
            col_min = self.df[column].min()
            col_max = self.df[column].max()
            if col_max != col_min:
                self.df[f'{column}_normalized'] = (self.df[column] - col_min) / (col_max - col_min)
            else:
                self.df[f'{column}_normalized'] = 0
        else:
            raise ValueError("Method must be 'zscore' or 'minmax'")
        
        return self.df[f'{column}_normalized']
    
    def fill_missing_with_strategy(self, columns=None, strategy='median'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns and self.df[col].isnull().any():
                if strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 0
                else:
                    raise ValueError("Strategy must be 'median', 'mean', or 'mode'")
                
                self.df[col] = self.df[col].fillna(fill_value)
        
        return self.df.isnull().sum().sum()
    
    def get_cleaned_data(self):
        return self.df.copy()
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'current_rows': self.df.shape[0],
            'columns': self.df.shape[1],
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns)
        }
        return summary

def example_usage():
    np.random.seed(42)
    data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'score': np.random.uniform(0, 1, 100)
    }
    
    df = pd.DataFrame(data)
    df.loc[10:15, 'value'] = np.nan
    df.loc[95:99, 'value'] = 500
    
    cleaner = DataCleaner(df)
    print("Initial summary:", cleaner.get_summary())
    
    removed = cleaner.remove_outliers_iqr(['value'], threshold=1.5)
    print(f"Removed {removed} outliers")
    
    missing_filled = cleaner.fill_missing_with_strategy(['value'], strategy='median')
    print(f"Filled {missing_filled} missing values")
    
    cleaner.normalize_column('score', method='minmax')
    
    print("Final summary:", cleaner.get_summary())
    return cleaner.get_cleaned_data()

if __name__ == "__main__":
    cleaned_df = example_usage()
    print("\nFirst 5 rows of cleaned data:")
    print(cleaned_df.head())import pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    """
    Remove rows with missing values from DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to check for missing values. 
                 If None, checks all columns.
    
    Returns:
        Cleaned DataFrame
    """
    if columns is None:
        columns = df.columns
    
    return df.dropna(subset=columns)

def fill_missing_with_mean(df, columns=None):
    """
    Fill missing values with column mean.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to fill. If None, fills all numeric columns.
    
    Returns:
        DataFrame with filled values
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_filled = df.copy()
    for col in columns:
        if col in df.columns:
            df_filled[col] = df[col].fillna(df[col].mean())
    
    return df_filled

def remove_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        df: pandas DataFrame
        columns: list of numeric columns to check for outliers.
                 If None, checks all numeric columns.
        threshold: IQR multiplier for outlier detection
    
    Returns:
        DataFrame without outliers
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to standardize. 
                 If None, standardizes all numeric columns.
    
    Returns:
        DataFrame with standardized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_standardized = df.copy()
    
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            if std_val > 0:
                df_standardized[col] = (df[col] - mean_val) / std_val
    
    return df_standardized

def get_data_summary(df):
    """
    Generate summary statistics for DataFrame.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'numeric_stats': df.describe().to_dict() if not df.select_dtypes(include=[np.number]).empty else {}
    }
    
    return summary