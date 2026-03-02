
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def detect_outliers_iqr(self, column):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        return outliers
    
    def remove_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[column].dropna()))
        self.df = self.df[(z_scores < threshold) | (self.df[column].isna())]
        return self
    
    def normalize_column(self, column, method='minmax'):
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            self.df[column] = (self.df[column] - mean_val) / std_val
        return self
    
    def fill_missing(self, column, strategy='mean'):
        if strategy == 'mean':
            fill_value = self.df[column].mean()
        elif strategy == 'median':
            fill_value = self.df[column].median()
        elif strategy == 'mode':
            fill_value = self.df[column].mode()[0]
        else:
            fill_value = strategy
            
        self.df[column].fillna(fill_value, inplace=True)
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]

def clean_dataset(df, config):
    cleaner = DataCleaner(df)
    
    for column in df.columns:
        if column in config.get('outlier_columns', []):
            cleaner.remove_outliers_zscore(column, threshold=config.get('zscore_threshold', 3))
        
        if column in config.get('normalize_columns', []):
            cleaner.normalize_column(column, method=config.get('normalize_method', 'minmax'))
        
        if column in config.get('fill_missing_columns', []):
            cleaner.fill_missing(column, strategy=config.get('fill_strategy', 'mean'))
    
    return cleaner.get_cleaned_data(), cleaner.get_removed_count()
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        
    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.numeric_columns
            
        clean_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                Q1 = clean_df[col].quantile(0.25)
                Q3 = clean_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]
        return clean_df
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.numeric_columns
            
        clean_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                z_scores = np.abs(stats.zscore(clean_df[col].dropna()))
                clean_df = clean_df[(z_scores < threshold) | clean_df[col].isna()]
        return clean_df
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
            
        normalized_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val != min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
        return normalized_df
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
            
        normalized_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                mean_val = normalized_df[col].mean()
                std_val = normalized_df[col].std()
                if std_val > 0:
                    normalized_df[col] = (normalized_df[col] - mean_val) / std_val
        return normalized_df
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
            
        filled_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                median_val = filled_df[col].median()
                filled_df[col] = filled_df[col].fillna(median_val)
        return filled_df
    
    def get_summary(self):
        summary = {
            'original_shape': self.df.shape,
            'numeric_columns': list(self.numeric_columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary
import pandas as pd
import numpy as np
from pathlib import Path

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None
        
    def load_data(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        self.df = pd.read_csv(self.file_path)
        print(f"Loaded data with shape: {self.df.shape}")
        return self.df
    
    def check_missing_values(self):
        if self.df is None:
            self.load_data()
        
        missing_counts = self.df.isnull().sum()
        missing_percentage = (missing_counts / len(self.df)) * 100
        
        missing_info = pd.DataFrame({
            'missing_count': missing_counts,
            'missing_percentage': missing_percentage
        })
        
        return missing_info[missing_info['missing_count'] > 0]
    
    def fill_missing_numeric(self, strategy='mean'):
        if self.df is None:
            self.load_data()
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'zero':
                    fill_value = 0
                else:
                    raise ValueError("Strategy must be 'mean', 'median', or 'zero'")
                
                self.df[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in {col} with {strategy}: {fill_value}")
        
        return self.df
    
    def fill_missing_categorical(self, strategy='mode'):
        if self.df is None:
            self.load_data()
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if self.df[col].isnull().any():
                if strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                elif strategy == 'unknown':
                    fill_value = 'Unknown'
                else:
                    raise ValueError("Strategy must be 'mode' or 'unknown'")
                
                self.df[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in {col} with {strategy}: {fill_value}")
        
        return self.df
    
    def drop_columns_with_high_missing(self, threshold=50):
        if self.df is None:
            self.load_data()
        
        missing_percentage = (self.df.isnull().sum() / len(self.df)) * 100
        columns_to_drop = missing_percentage[missing_percentage > threshold].index.tolist()
        
        if columns_to_drop:
            self.df.drop(columns=columns_to_drop, inplace=True)
            print(f"Dropped columns with >{threshold}% missing values: {columns_to_drop}")
        
        return self.df
    
    def save_cleaned_data(self, output_path=None):
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        self.df.to_csv(output_path, index=False)
        print(f"Saved cleaned data to: {output_path}")
        return output_path

def process_csv_file(input_file, output_file=None):
    cleaner = DataCleaner(input_file)
    
    cleaner.load_data()
    
    missing_report = cleaner.check_missing_values()
    print("Missing values report:")
    print(missing_report)
    
    cleaner.fill_missing_numeric(strategy='mean')
    cleaner.fill_missing_categorical(strategy='mode')
    cleaner.drop_columns_with_high_missing(threshold=50)
    
    output_path = cleaner.save_cleaned_data(output_file)
    
    return cleaner.df, output_path

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', None, 'David', 'Eve'],
        'age': [25, 30, None, 35, None],
        'score': [85.5, 92.0, 78.5, None, 88.0],
        'department': ['HR', 'IT', 'IT', None, 'HR']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_file = Path('test_data.csv')
    test_df.to_csv(test_file, index=False)
    
    cleaned_df, output_file = process_csv_file(test_file)
    
    print("\nCleaned data preview:")
    print(cleaned_df.head())
    
    test_file.unlink()
    Path(output_file).unlink()