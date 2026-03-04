
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
    Path(output_file).unlink()import csv
import os
from typing import List, Dict, Any

def read_csv_file(file_path: str) -> List[Dict[str, Any]]:
    """Read a CSV file and return its contents as a list of dictionaries."""
    data = []
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
    return data

def clean_numeric_fields(data: List[Dict[str, Any]], fields: List[str]) -> List[Dict[str, Any]]:
    """Clean specified numeric fields by removing non-numeric characters and converting to float."""
    cleaned_data = []
    for row in data:
        cleaned_row = row.copy()
        for field in fields:
            if field in cleaned_row:
                value = cleaned_row[field]
                if isinstance(value, str):
                    cleaned_value = ''.join(char for char in value if char.isdigit() or char == '.')
                    try:
                        cleaned_row[field] = float(cleaned_value) if cleaned_value else 0.0
                    except ValueError:
                        cleaned_row[field] = 0.0
        cleaned_data.append(cleaned_row)
    return cleaned_data

def remove_empty_rows(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove rows where all values are empty strings or None."""
    filtered_data = []
    for row in data:
        if any(value not in [None, ''] for value in row.values()):
            filtered_data.append(row)
    return filtered_data

def write_csv_file(data: List[Dict[str, Any]], file_path: str) -> bool:
    """Write data to a CSV file."""
    if not data:
        print("No data to write.")
        return False
    
    try:
        fieldnames = data[0].keys()
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        return True
    except Exception as e:
        print(f"Error writing CSV file: {e}")
        return False

def process_csv(input_path: str, output_path: str, numeric_fields: List[str] = None) -> None:
    """Main function to process a CSV file: read, clean, and write."""
    if numeric_fields is None:
        numeric_fields = []
    
    print(f"Processing file: {input_path}")
    data = read_csv_file(input_path)
    
    if not data:
        print("No data loaded. Exiting.")
        return
    
    print(f"Loaded {len(data)} rows.")
    
    cleaned_data = clean_numeric_fields(data, numeric_fields)
    cleaned_data = remove_empty_rows(cleaned_data)
    
    print(f"After cleaning: {len(cleaned_data)} rows.")
    
    if write_csv_file(cleaned_data, output_path):
        print(f"Cleaned data saved to: {output_path}")
    else:
        print("Failed to save cleaned data.")

if __name__ == "__main__":
    input_file = "input_data.csv"
    output_file = "cleaned_data.csv"
    numeric_columns = ["price", "quantity", "rating"]
    
    if os.path.exists(input_file):
        process_csv(input_file, output_file, numeric_columns)
    else:
        print(f"Input file '{input_file}' does not exist. Creating sample data for testing.")
        sample_data = [
            {"id": "1", "name": "Product A", "price": "$19.99", "quantity": "10", "rating": "4.5"},
            {"id": "2", "name": "Product B", "price": "29.50", "quantity": "", "rating": "3.8"},
            {"id": "3", "name": "", "price": "invalid", "quantity": "5", "rating": "4.0"},
            {"id": "4", "name": "Product D", "price": "15.00", "quantity": "7", "rating": ""},
            {"id": "", "name": "", "price": "", "quantity": "", "rating": ""}
        ]
        
        with open(input_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["id", "name", "price", "quantity", "rating"])
            writer.writeheader()
            writer.writerows(sample_data)
        
        print(f"Sample data created in '{input_file}'. Now processing...")
        process_csv(input_file, output_file, numeric_columns)
import pandas as pd

def remove_duplicates(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        dataframe: pandas DataFrame to process
        subset: column label or sequence of labels to consider for duplicates
        keep: determines which duplicates to keep ('first', 'last', False)
    
    Returns:
        DataFrame with duplicates removed
    """
    if dataframe.empty:
        return dataframe
    
    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(dataframe) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate row(s)")
    
    return cleaned_df

def clean_numeric_columns(dataframe, columns):
    """
    Clean numeric columns by removing non-numeric values and converting to float.
    
    Args:
        dataframe: pandas DataFrame to process
        columns: list of column names to clean
    
    Returns:
        DataFrame with cleaned numeric columns
    """
    for column in columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors='coerce')
    
    return dataframe

def validate_dataframe(dataframe, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe: pandas DataFrame to validate
        required_columns: list of required column names
    
    Returns:
        Tuple of (is_valid, message)
    """
    if dataframe.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"