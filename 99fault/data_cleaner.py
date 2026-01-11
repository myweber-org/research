
import pandas as pd
import numpy as np
from pathlib import Path

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None
        
    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Loaded {len(self.df)} rows from {self.file_path.name}")
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns and self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
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
    
    def normalize_column(self, column_name):
        if column_name in self.df.columns and self.df[column_name].dtype in [np.float64, np.int64]:
            col_min = self.df[column_name].min()
            col_max = self.df[column_name].max()
            
            if col_max != col_min:
                self.df[column_name] = (self.df[column_name] - col_min) / (col_max - col_min)
                print(f"Normalized column '{column_name}' to range [0, 1]")
            else:
                print(f"Column '{column_name}' has constant values, skipping normalization")
    
    def save_cleaned_data(self, output_path=None):
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        self.df.to_csv(output_path, index=False)
        print(f"Saved cleaned data to {output_path}")
        return output_path
    
    def get_summary(self):
        if self.df is None:
            return "No data loaded"
        
        summary = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicate_rows': self.df.duplicated().sum(),
            'data_types': self.df.dtypes.value_counts().to_dict()
        }
        
        return summary

def process_csv_file(input_file, output_dir='cleaned_data'):
    cleaner = DataCleaner(input_file)
    
    if cleaner.load_data():
        print("Initial data summary:", cleaner.get_summary())
        
        cleaner.handle_missing_values(strategy='mean')
        cleaner.remove_duplicates()
        
        numeric_cols = cleaner.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:3]:
            cleaner.normalize_column(col)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        output_path = cleaner.save_cleaned_data(output_dir / f"cleaned_{Path(input_file).name}")
        
        print("Final data summary:", cleaner.get_summary())
        return output_path
    
    return None

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, None, 15.2, 20.1, None, 10.5],
        'category': ['A', 'B', 'A', 'C', 'B', 'A'],
        'score': [85, 92, 78, None, 88, 85]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_file = "test_data.csv"
    test_df.to_csv(test_file, index=False)
    
    result = process_csv_file(test_file)
    
    if result:
        print(f"Processing complete. Cleaned file saved to: {result}")
    
    Path(test_file).unlink(missing_ok=True)
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column in a DataFrame using the IQR method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from the specified column.
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
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing count, mean, std, min, and max.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'count': df[column].count(),
        'mean': df[column].mean(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max()
    }
    
    return stats

def process_dataset(file_path, column_to_clean):
    """
    Load a dataset from a CSV file and clean a specified column.
    
    Parameters:
    file_path (str): Path to the CSV file.
    column_to_clean (str): Column name to clean.
    
    Returns:
    tuple: (cleaned DataFrame, original stats, cleaned stats)
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    original_stats = calculate_basic_stats(df, column_to_clean)
    cleaned_df = remove_outliers_iqr(df, column_to_clean)
    cleaned_stats = calculate_basic_stats(cleaned_df, column_to_clean)
    
    return cleaned_df, original_stats, cleaned_stats

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'values': np.concatenate([
            np.random.normal(100, 15, 95),
            np.random.normal(300, 50, 5)
        ])
    })
    
    print("Original data shape:", sample_data.shape)
    print("Original statistics:", calculate_basic_stats(sample_data, 'values'))
    
    cleaned_data = remove_outliers_iqr(sample_data, 'values')
    print("Cleaned data shape:", cleaned_data.shape)
    print("Cleaned statistics:", calculate_basic_stats(cleaned_data, 'values'))
import pandas as pd
import re

def clean_dataset(df, columns_to_clean=None):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_clean (list, optional): List of column names to apply text normalization.
            If None, all object dtype columns are cleaned.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    removed_duplicates = initial_rows - len(cleaned_df)
    
    # Normalize text columns
    if columns_to_clean is None:
        text_columns = cleaned_df.select_dtypes(include=['object']).columns
    else:
        text_columns = [col for col in columns_to_clean if col in cleaned_df.columns]
    
    for col in text_columns:
        if cleaned_df[col].dtype == 'object':
            cleaned_df[col] = cleaned_df[col].apply(_normalize_text)
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    print(f"Removed {removed_duplicates} duplicate rows.")
    print(f"Normalized text in columns: {list(text_columns)}")
    
    return cleaned_df

def _normalize_text(text):
    """
    Normalize text by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    
    Args:
        text (str): Input text to normalize.
    
    Returns:
        str: Normalized text.
    """
    if pd.isna(text):
        return text
    
    # Convert to string if not already
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters (keep alphanumeric and basic punctuation)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    return text

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        email_column (str): Name of the column containing email addresses.
    
    Returns:
        pd.DataFrame: DataFrame with additional 'email_valid' column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame.")
    
    validated_df = df.copy()
    
    # Simple email validation regex
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    validated_df['email_valid'] = validated_df[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notna(x) else False
    )
    
    valid_count = validated_df['email_valid'].sum()
    total_count = len(validated_df)
    
    print(f"Email validation: {valid_count}/{total_count} valid emails ({valid_count/total_count*100:.1f}%)")
    
    return validated_df
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
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
    
    return filtered_df

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
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
        'count': df[column].count()
    }
    
    return stats

def process_dataset(df, numeric_columns):
    """
    Process dataset by removing outliers from multiple numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of column names to process
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of statistics for each column
    """
    cleaned_df = df.copy()
    all_stats = {}
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_summary_statistics(cleaned_df, column)
            stats['outliers_removed'] = removed_count
            all_stats[column] = stats
    
    return cleaned_df, all_stats

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 27, 28, 29, 30, 100],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 53, 150],
        'pressure': [1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 2000]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    numeric_cols = ['temperature', 'humidity', 'pressure']
    cleaned_df, statistics = process_dataset(df, numeric_cols)
    
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    print("Summary Statistics:")
    for col, stats in statistics.items():
        print(f"\n{col}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")