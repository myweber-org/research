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