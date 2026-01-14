
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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
    
    return filtered_df.copy()

def clean_dataset(df, columns_to_clean=None):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns_to_clean is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns_to_clean = list(numeric_cols)
    
    cleaned_df = df.copy()
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            original_len = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_len - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_dfimport pandas as pd
import numpy as np
from pathlib import Path

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None
        
    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Loaded data with shape: {self.df.shape}")
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        if columns is None:
            columns = self.df.columns
        
        for col in columns:
            if col in self.df.columns and self.df[col].isnull().any():
                if strategy == 'mean' and pd.api.types.is_numeric_dtype(self.df[col]):
                    fill_value = self.df[col].mean()
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(self.df[col]):
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                elif strategy == 'drop':
                    self.df = self.df.dropna(subset=[col])
                    continue
                else:
                    fill_value = 0
                
                self.df[col] = self.df[col].fillna(fill_value)
                print(f"Filled missing values in '{col}' using {strategy} strategy")
    
    def remove_duplicates(self, subset=None, keep='first'):
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed = initial_count - len(self.df)
        print(f"Removed {removed} duplicate rows")
    
    def normalize_numeric(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                    print(f"Normalized column '{col}' to range [0, 1]")
    
    def save_cleaned_data(self, output_path=None):
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        self.df.to_csv(output_path, index=False)
        print(f"Saved cleaned data to: {output_path}")
        return output_path
    
    def get_summary(self):
        if self.df is None:
            return "No data available"
        
        summary = {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicates': len(self.df) - len(self.df.drop_duplicates()),
            'numeric_columns': len(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(self.df.select_dtypes(include=['object']).columns)
        }
        
        return summary

def clean_csv_file(input_file, output_file=None, missing_strategy='mean'):
    cleaner = DataCleaner(input_file)
    
    if cleaner.load_data():
        print("Starting data cleaning process...")
        cleaner.handle_missing_values(strategy=missing_strategy)
        cleaner.remove_duplicates()
        cleaner.normalize_numeric()
        
        saved_path = cleaner.save_cleaned_data(output_file)
        
        summary = cleaner.get_summary()
        print("\nCleaning Summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        return saved_path
    else:
        return None

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'age': [25, np.nan, 30, 35, 40, 40],
        'salary': [50000, 60000, np.nan, 80000, 90000, 90000],
        'department': ['IT', 'HR', 'IT', 'Finance', 'HR', 'HR']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_file = Path('test_data.csv')
    test_df.to_csv(test_file, index=False)
    
    cleaned_file = clean_csv_file(test_file, missing_strategy='mean')
    
    if cleaned_file:
        print(f"\nCleaned file saved at: {cleaned_file}")
        cleaned_df = pd.read_csv(cleaned_file)
        print("\nFirst few rows of cleaned data:")
        print(cleaned_df.head())
    
    test_file.unlink()
    if cleaned_file and Path(cleaned_file).exists():
        Path(cleaned_file).unlink()