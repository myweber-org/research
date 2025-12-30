
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
        
        if self.file_path.suffix == '.csv':
            self.df = pd.read_csv(self.file_path)
        elif self.file_path.suffix in ['.xlsx', '.xls']:
            self.df = pd.read_excel(self.file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel files.")
        
        return self.df
    
    def remove_duplicates(self):
        if self.df is None:
            self.load_data()
        
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = initial_rows - len(self.df)
        
        return removed
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if self.df is None:
            self.load_data()
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for column in columns:
            if column not in self.df.columns:
                continue
                
            if strategy == 'mean':
                fill_value = self.df[column].mean()
            elif strategy == 'median':
                fill_value = self.df[column].median()
            elif strategy == 'mode':
                fill_value = self.df[column].mode()[0]
            elif strategy == 'zero':
                fill_value = 0
            else:
                raise ValueError("Invalid strategy. Use 'mean', 'median', 'mode', or 'zero'")
            
            self.df[column] = self.df[column].fillna(fill_value)
        
        return self.df
    
    def normalize_column(self, column_name):
        if self.df is None:
            self.load_data()
        
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in dataframe")
        
        col_min = self.df[column_name].min()
        col_max = self.df[column_name].max()
        
        if col_max == col_min:
            self.df[column_name] = 0
        else:
            self.df[column_name] = (self.df[column_name] - col_min) / (col_max - col_min)
        
        return self.df
    
    def save_cleaned_data(self, output_path=None):
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        output_path = Path(output_path)
        
        if output_path.suffix == '.csv':
            self.df.to_csv(output_path, index=False)
        elif output_path.suffix in ['.xlsx', '.xls']:
            self.df.to_excel(output_path, index=False)
        else:
            raise ValueError("Unsupported output format. Use CSV or Excel files.")
        
        return output_path
    
    def get_summary(self):
        if self.df is None:
            self.load_data()
        
        summary = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicate_rows': self.df.duplicated().sum(),
            'data_types': self.df.dtypes.to_dict(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns)
        }
        
        return summary

def clean_dataset(input_file, output_file=None):
    cleaner = DataCleaner(input_file)
    cleaner.load_data()
    
    duplicates_removed = cleaner.remove_duplicates()
    print(f"Removed {duplicates_removed} duplicate rows")
    
    cleaner.handle_missing_values(strategy='mean')
    print("Handled missing values using mean imputation")
    
    summary = cleaner.get_summary()
    print(f"Dataset summary: {summary['total_rows']} rows, {summary['total_columns']} columns")
    
    output_path = cleaner.save_cleaned_data(output_file)
    print(f"Cleaned data saved to: {output_path}")
    
    return output_path
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

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            original_len = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_len - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 101),
        'value': np.concatenate([
            np.random.normal(100, 10, 90),
            np.random.normal(300, 50, 10)
        ]),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame shape:", df.shape)
    
    stats_before = calculate_basic_stats(df, 'value')
    print("\nStatistics before cleaning:")
    for key, value in stats_before.items():
        print(f"{key}: {value:.2f}")
    
    cleaned_df = clean_numeric_data(df, ['value'])
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    
    stats_after = calculate_basic_stats(cleaned_df, 'value')
    print("\nStatistics after cleaning:")
    for key, value in stats_after.items():
        print(f"{key}: {value:.2f}")