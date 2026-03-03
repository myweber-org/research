import pandas as pd
import numpy as np
from pathlib import Path

class DataCleaner:
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.df = None
        
    def load_data(self):
        try:
            self.df = pd.read_csv(self.filepath)
            print(f"Loaded data with shape: {self.df.shape}")
            return True
        except FileNotFoundError:
            print(f"Error: File not found at {self.filepath}")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def remove_duplicates(self):
        if self.df is not None:
            initial_count = len(self.df)
            self.df.drop_duplicates(inplace=True)
            removed = initial_count - len(self.df)
            print(f"Removed {removed} duplicate rows")
            return removed
        return 0
    
    def fill_missing_values(self, strategy='mean', columns=None):
        if self.df is not None:
            if columns is None:
                columns = self.df.select_dtypes(include=[np.number]).columns
            
            for col in columns:
                if col in self.df.columns:
                    if strategy == 'mean':
                        fill_value = self.df[col].mean()
                    elif strategy == 'median':
                        fill_value = self.df[col].median()
                    elif strategy == 'mode':
                        fill_value = self.df[col].mode()[0]
                    else:
                        fill_value = 0
                    
                    missing_count = self.df[col].isnull().sum()
                    self.df[col].fillna(fill_value, inplace=True)
                    print(f"Filled {missing_count} missing values in column '{col}'")
    
    def remove_outliers(self, columns=None, threshold=3):
        if self.df is not None:
            if columns is None:
                columns = self.df.select_dtypes(include=[np.number]).columns
            
            initial_count = len(self.df)
            for col in columns:
                if col in self.df.columns:
                    z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                    self.df = self.df[z_scores < threshold]
            
            removed = initial_count - len(self.df)
            print(f"Removed {removed} outliers")
            return removed
        return 0
    
    def standardize_columns(self, columns=None):
        if self.df is not None:
            if columns is None:
                columns = self.df.select_dtypes(include=[np.number]).columns
            
            for col in columns:
                if col in self.df.columns:
                    mean = self.df[col].mean()
                    std = self.df[col].std()
                    if std > 0:
                        self.df[col] = (self.df[col] - mean) / std
                        print(f"Standardized column '{col}'")
    
    def save_cleaned_data(self, output_path=None):
        if self.df is not None:
            if output_path is None:
                output_path = self.filepath.parent / f"cleaned_{self.filepath.name}"
            
            self.df.to_csv(output_path, index=False)
            print(f"Saved cleaned data to {output_path}")
            return output_path
        return None
    
    def get_summary(self):
        if self.df is not None:
            summary = {
                'rows': len(self.df),
                'columns': len(self.df.columns),
                'missing_values': self.df.isnull().sum().sum(),
                'duplicates': self.df.duplicated().sum(),
                'data_types': self.df.dtypes.to_dict()
            }
            return summary
        return {}

def clean_csv_file(input_file, output_file=None):
    cleaner = DataCleaner(input_file)
    
    if cleaner.load_data():
        print("Starting data cleaning process...")
        
        cleaner.remove_duplicates()
        cleaner.fill_missing_values(strategy='mean')
        cleaner.remove_outliers()
        cleaner.standardize_columns()
        
        output_path = cleaner.save_cleaned_data(output_file)
        summary = cleaner.get_summary()
        
        print("\nCleaning Summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        return output_path
    return None

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, 4, 5, 100],
        'B': [10, 20, 20, 30, np.nan, 50, 60],
        'C': ['x', 'y', 'y', 'z', 'z', 'x', 'x']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    result = clean_csv_file('test_data.csv')
    print(f"\nCleaned file saved at: {result}")import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    if max_val - min_val == 0:
        return df[column]
    return (df[column] - min_val) / (max_val - min_val)

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df

def calculate_statistics(df):
    stats = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        stats[col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'median': df[col].median()
        }
    return pd.DataFrame(stats).T

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    })
    
    cleaned = clean_dataset(sample_data, ['A', 'B', 'C'])
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    print("\nStatistics:")
    print(calculate_statistics(cleaned))
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
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                df_clean = df_clean[mask]
        
        self.df = df_clean.reset_index(drop=True)
        return self
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_clean = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                z_scores = np.abs(stats.zscore(self.df[col].fillna(self.df[col].mean())))
                mask = z_scores < threshold
                df_clean = df_clean[mask]
        
        self.df = df_clean.reset_index(drop=True)
        return self
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    df_normalized[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        self.df = df_normalized
        return self
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    df_normalized[col] = (self.df[col] - mean_val) / std_val
        
        self.df = df_normalized
        return self
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_filled = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                df_filled[col] = self.df[col].fillna(self.df[col].mean())
        
        self.df = df_filled
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'rows_removed': self.get_removed_count(),
            'removal_percentage': (self.get_removed_count() / self.original_shape[0]) * 100 if self.original_shape[0] > 0 else 0,
            'columns': list(self.df.columns)
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
    
    data['feature_a'][np.random.choice(1000, 10)] = np.nan
    data['feature_b'][np.random.choice(1000, 20)] = np.nan
    
    outliers = np.random.choice(1000, 5)
    data['feature_a'][outliers] = 500
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = create_sample_data()
    print("Original data shape:", df.shape)
    print("Missing values:\n", df.isnull().sum())
    
    cleaner = DataCleaner(df)
    cleaner.fill_missing_mean().remove_outliers_iqr(factor=1.5).normalize_minmax()
    
    cleaned_df = cleaner.get_cleaned_data()
    print("\nCleaned data shape:", cleaned_df.shape)
    print("Summary:", cleaner.get_summary())
    print("\nFirst 5 rows of cleaned data:")
    print(cleaned_df.head())import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset (list, optional): Columns to consider for duplicates
    keep (str): Which duplicates to keep - 'first', 'last', or False
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_column(df, column_name, fill_method='mean'):
    """
    Clean numeric column by handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column_name (str): Name of column to clean
    fill_method (str): Method to fill missing values - 'mean', 'median', or 'zero'
    
    Returns:
    pd.DataFrame: DataFrame with cleaned column
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        raise ValueError(f"Column '{column_name}' is not numeric")
    
    df_clean = df.copy()
    
    missing_count = df_clean[column_name].isna().sum()
    if missing_count > 0:
        if fill_method == 'mean':
            fill_value = df_clean[column_name].mean()
        elif fill_method == 'median':
            fill_value = df_clean[column_name].median()
        elif fill_method == 'zero':
            fill_value = 0
        else:
            raise ValueError("fill_method must be 'mean', 'median', or 'zero'")
        
        df_clean[column_name] = df_clean[column_name].fillna(fill_value)
        print(f"Filled {missing_count} missing values in '{column_name}' with {fill_method}")
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list, optional): List of required column names
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

def get_data_summary(df):
    """
    Generate summary statistics for DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if df.empty:
        return {"row_count": 0, "column_count": 0}
    
    summary = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "column_names": list(df.columns),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isna().sum().to_dict(),
        "numeric_summary": {}
    }
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        summary["numeric_summary"][col] = {
            "mean": df[col].mean(),
            "median": df[col].median(),
            "std": df[col].std(),
            "min": df[col].min(),
            "max": df[col].max()
        }
    
    return summary