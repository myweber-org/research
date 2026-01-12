
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
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
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
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
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
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
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
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    df_normalized[col] = (self.df[col] - mean_val) / std_val
        
        self.df = df_normalized
        return self
        
    def fill_missing(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_filled = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 0
                else:
                    fill_value = 0
                
                df_filled[col] = self.df[col].fillna(fill_value)
        
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
            'columns': self.df.shape[1],
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
    
    indices = np.random.choice(df.index, size=50, replace=False)
    df.loc[indices, 'feature_a'] = np.random.uniform(200, 300, 50)
    
    missing_indices = np.random.choice(df.index, size=30, replace=False)
    df.loc[missing_indices, 'feature_b'] = np.nan
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Original data shape:", sample_df.shape)
    print("Missing values:", sample_df.isnull().sum().sum())
    
    cleaner = DataCleaner(sample_df)
    
    cleaner.remove_outliers_iqr(['feature_a', 'feature_b'])
    cleaner.fill_missing(strategy='median')
    cleaner.normalize_minmax(['feature_a', 'feature_b', 'feature_c'])
    
    cleaned_df = cleaner.get_cleaned_data()
    summary = cleaner.get_summary()
    
    print("\nCleaning Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\nCleaned data shape:", cleaned_df.shape)
    print("Cleaned data columns:", cleaned_df.columns.tolist())
    print("\nFirst 5 rows of cleaned data:")
    print(cleaned_df.head())
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. 
                        Options: 'mean', 'median', 'mode', 'drop', or 'zero'.
                        Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values.")
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if fill_missing == 'mean':
                fill_value = cleaned_df[col].mean()
            elif fill_missing == 'median':
                fill_value = cleaned_df[col].median()
            elif fill_missing == 'mode':
                fill_value = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 0
            
            missing_count = cleaned_df[col].isna().sum()
            if missing_count > 0:
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                print(f"Filled {missing_count} missing values in column '{col}' with {fill_missing}: {fill_value:.2f}")
    elif fill_missing == 'zero':
        cleaned_df = cleaned_df.fillna(0)
        print("Filled missing values with zero.")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (is_valid, message)
    """
    if df.empty:
        return False, "DataFrame is empty."
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows."
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid."

def remove_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range (IQR) method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to check for outliers.
                    If None, uses all numeric columns.
    multiplier (float): IQR multiplier for outlier detection.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    for col in columns:
        if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            outliers_removed = len(df_clean) - mask.sum()
            
            if outliers_removed > 0:
                df_clean = df_clean[mask]
                print(f"Removed {outliers_removed} outliers from column '{col}'")
    
    final_rows = len(df_clean)
    total_removed = initial_rows - final_rows
    
    if total_removed > 0:
        print(f"Total outliers removed: {total_removed}")
    
    return df_clean

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize numeric columns in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to normalize.
                    If None, uses all numeric columns.
    method (str): Normalization method. Options: 'minmax', 'zscore'.
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    
    for col in columns:
        if col in df_normalized.columns and pd.api.types.is_numeric_dtype(df_normalized[col]):
            if method == 'minmax':
                min_val = df_normalized[col].min()
                max_val = df_normalized[col].max()
                if max_val > min_val:
                    df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = df_normalized[col].mean()
                std_val = df_normalized[col].std()
                if std_val > 0:
                    df_normalized[col] = (df_normalized[col] - mean_val) / std_val
    
    return df_normalized

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, 5, np.nan, 7, 8, 9, 100],
        'B': [10, 20, 20, 40, 50, 60, 70, 80, 90, 1000],
        'C': ['a', 'b', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataframe(df, fill_missing='median')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {message}")
    
    no_outliers = remove_outliers_iqr(cleaned, columns=['A', 'B'])
    print("\nDataFrame after outlier removal:")
    print(no_outliers)
    
    normalized = normalize_data(no_outliers, method='minmax')
    print("\nNormalized DataFrame:")
    print(normalized)