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
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        self.df = df_clean
        removed = self.original_shape[0] - self.df.shape[0]
        print(f"Removed {removed} outliers using IQR method")
        return self
        
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        print("Applied Min-Max normalization")
        return self
        
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
        
        print("Applied Z-score standardization")
        return self
        
    def handle_missing_mean(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                mean_val = self.df[col].mean()
                self.df[col] = self.df[col].fillna(mean_val)
        
        print("Filled missing values with column means")
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
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns)
        }
        return summary

def load_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 200, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 50), 'feature_a'] = np.nan
    df.loc[np.random.choice(df.index, 20), 'feature_b'] = 1000
    return df

if __name__ == "__main__":
    sample_df = load_sample_data()
    cleaner = DataCleaner(sample_df)
    
    cleaned_df = (cleaner
                 .handle_missing_mean()
                 .remove_outliers_iqr(factor=1.5)
                 .standardize_zscore()
                 .get_cleaned_data())
    
    summary = cleaner.get_summary()
    print(f"Data cleaning complete. Removed {summary['rows_removed']} rows.")
    print(f"Final shape: {cleaned_df.shape}")
import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean', columns_to_drop=None):
    """
    Load a CSV file, clean missing values, and optionally drop specified columns.
    
    Args:
        filepath (str): Path to the CSV file.
        fill_strategy (str): Strategy for filling missing values. 
                             Options: 'mean', 'median', 'mode', 'zero'.
        columns_to_drop (list): List of column names to drop from the dataset.
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    original_shape = df.shape
    
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors='ignore')
    
    if fill_strategy == 'mean':
        df = df.fillna(df.mean(numeric_only=True))
    elif fill_strategy == 'median':
        df = df.fillna(df.median(numeric_only=True))
    elif fill_strategy == 'mode':
        df = df.fillna(df.mode().iloc[0])
    elif fill_strategy == 'zero':
        df = df.fillna(0)
    else:
        print(f"Warning: Unknown fill strategy '{fill_strategy}'. Using 'mean'.")
        df = df.fillna(df.mean(numeric_only=True))
    
    df = df.dropna()
    
    print(f"Data cleaning completed:")
    print(f"  Original shape: {original_shape}")
    print(f"  Final shape: {df.shape}")
    print(f"  Rows removed: {original_shape[0] - df.shape[0]}")
    print(f"  Columns removed: {original_shape[1] - df.shape[1]}")
    
    return df

def save_cleaned_data(df, output_path):
    """
    Save the cleaned DataFrame to a CSV file.
    
    Args:
        df (pandas.DataFrame): DataFrame to save.
        output_path (str): Path for the output CSV file.
    """
    if df is not None:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
    else:
        print("Error: No data to save.")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(
        filepath=input_file,
        fill_strategy='median',
        columns_to_drop=['unnecessary_column', 'old_id']
    )
    
    if cleaned_df is not None:
        save_cleaned_data(cleaned_df, output_file)