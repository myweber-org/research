
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
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
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Args:
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
        'count': len(df[column])
    }
    
    return stats

def process_dataframe(df, numeric_columns):
    """
    Process multiple numeric columns to remove outliers.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        numeric_columns (list): List of numeric column names to process
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[column]):
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[::100, 'A'] = 500
    
    print("Original DataFrame shape:", df.shape)
    print("Original statistics for column A:")
    print(calculate_summary_statistics(df, 'A'))
    
    cleaned_df = process_dataframe(df, ['A', 'B', 'C'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned statistics for column A:")
    print(calculate_summary_statistics(cleaned_df, 'A'))
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_columns = df.columns.tolist()
    
    def remove_outliers_iqr(self, column, multiplier=1.5):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        return self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
    
    def remove_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[column]))
        return self.df[z_scores < threshold]
    
    def normalize_minmax(self, column):
        min_val = self.df[column].min()
        max_val = self.df[column].max()
        self.df[f'{column}_normalized'] = (self.df[column] - min_val) / (max_val - min_val)
        return self.df
    
    def standardize_zscore(self, column):
        mean_val = self.df[column].mean()
        std_val = self.df[column].std()
        self.df[f'{column}_standardized'] = (self.df[column] - mean_val) / std_val
        return self.df
    
    def handle_missing_mean(self, column):
        self.df[column].fillna(self.df[column].mean(), inplace=True)
        return self.df
    
    def handle_missing_median(self, column):
        self.df[column].fillna(self.df[column].median(), inplace=True)
        return self.df
    
    def get_summary(self):
        summary = {
            'original_shape': self.df.shape,
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict(),
            'numeric_columns': self.df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': self.df.select_dtypes(include=['object']).columns.tolist()
        }
        return summary
    
    def save_cleaned_data(self, filepath):
        self.df.to_csv(filepath, index=False)
        print(f"Cleaned data saved to {filepath}")

def example_usage():
    data = {
        'age': [25, 30, 35, 200, 28, 32, 150, 29, 31, 33],
        'salary': [50000, 60000, 70000, 1000000, 55000, 65000, 2000000, 58000, 62000, 68000],
        'department': ['IT', 'HR', 'IT', 'Finance', 'HR', 'IT', 'Finance', 'HR', 'IT', 'Finance']
    }
    
    df = pd.DataFrame(data)
    cleaner = DataCleaner(df)
    
    print("Original data summary:")
    print(cleaner.get_summary())
    
    cleaner.remove_outliers_iqr('age')
    cleaner.remove_outliers_zscore('salary')
    cleaner.normalize_minmax('age')
    cleaner.standardize_zscore('salary')
    
    print("\nCleaned data columns:", cleaner.df.columns.tolist())
    return cleaner.df

if __name__ == "__main__":
    cleaned_df = example_usage()
    print(f"\nFinal shape: {cleaned_df.shape}")
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from the specified column.
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
    Calculate summary statistics for a DataFrame column.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing count, mean, std, min, max.
    """
    stats = {
        'count': data[column].count(),
        'mean': data[column].mean(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max()
    }
    return stats