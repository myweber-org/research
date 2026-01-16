
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_summary_statistics(data, column):
    """
    Calculate summary statistics for a specified column.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std()
    }
    return stats

if __name__ == "__main__":
    import pandas as pd
    
    sample_data = pd.DataFrame({
        'values': np.random.normal(100, 15, 1000)
    })
    
    cleaned_data = remove_outliers_iqr(sample_data, 'values')
    stats = calculate_summary_statistics(cleaned_data, 'values')
    
    print(f"Original data shape: {sample_data.shape}")
    print(f"Cleaned data shape: {cleaned_data.shape}")
    print(f"Summary statistics: {stats}")
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, column, multiplier=1.5):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        return self
        
    def remove_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[column]))
        self.df = self.df[z_scores < threshold]
        return self
        
    def normalize_column(self, column, method='minmax'):
        if method == 'minmax':
            self.df[column] = (self.df[column] - self.df[column].min()) / (self.df[column].max() - self.df[column].min())
        elif method == 'zscore':
            self.df[column] = (self.df[column] - self.df[column].mean()) / self.df[column].std()
        return self
        
    def fill_missing(self, column, method='mean'):
        if method == 'mean':
            self.df[column].fillna(self.df[column].mean(), inplace=True)
        elif method == 'median':
            self.df[column].fillna(self.df[column].median(), inplace=True)
        elif method == 'mode':
            self.df[column].fillna(self.df[column].mode()[0], inplace=True)
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]import pandas as pd
import numpy as np

def remove_duplicates(df):
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates()

def fill_missing_values(df, strategy='mean'):
    """
    Fill missing values in numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == 'zero':
        df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df

def normalize_columns(df, columns=None):
    """
    Normalize specified columns to range [0, 1].
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_max != col_min:
                df[col] = (df[col] - col_min) / (col_max - col_min)
    
    return df

def clean_dataframe(df, remove_dups=True, fill_na=True, normalize=True):
    """
    Apply multiple cleaning operations to DataFrame.
    """
    if remove_dups:
        df = remove_duplicates(df)
    
    if fill_na:
        df = fill_missing_values(df)
    
    if normalize:
        df = normalize_columns(df)
    
    return df

def load_and_clean_csv(filepath):
    """
    Load CSV file and apply cleaning operations.
    """
    try:
        df = pd.read_csv(filepath)
        return clean_dataframe(df)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None