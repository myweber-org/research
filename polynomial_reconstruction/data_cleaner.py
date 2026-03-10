
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, data):
        self.data = data
        self.original_shape = data.shape
        
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.data.columns
            
        clean_data = self.data.copy()
        for col in columns:
            if clean_data[col].dtype in ['int64', 'float64']:
                Q1 = clean_data[col].quantile(0.25)
                Q3 = clean_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                clean_data = clean_data[(clean_data[col] >= lower_bound) & 
                                       (clean_data[col] <= upper_bound)]
        return clean_data
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        normalized_data = self.data.copy()
        for col in columns:
            if normalized_data[col].dtype in ['int64', 'float64']:
                min_val = normalized_data[col].min()
                max_val = normalized_data[col].max()
                if max_val > min_val:
                    normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
        return normalized_data
    
    def standardize_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        standardized_data = self.data.copy()
        for col in columns:
            if standardized_data[col].dtype in ['int64', 'float64']:
                mean_val = standardized_data[col].mean()
                std_val = standardized_data[col].std()
                if std_val > 0:
                    z_scores = np.abs((standardized_data[col] - mean_val) / std_val)
                    standardized_data = standardized_data[z_scores < threshold]
        return standardized_data
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.data.columns
            
        filled_data = self.data.copy()
        for col in columns:
            if filled_data[col].isnull().any():
                if strategy == 'mean' and filled_data[col].dtype in ['int64', 'float64']:
                    fill_value = filled_data[col].mean()
                elif strategy == 'median' and filled_data[col].dtype in ['int64', 'float64']:
                    fill_value = filled_data[col].median()
                elif strategy == 'mode':
                    fill_value = filled_data[col].mode()[0]
                elif strategy == 'drop':
                    filled_data = filled_data.dropna(subset=[col])
                    continue
                else:
                    fill_value = 0
                filled_data[col] = filled_data[col].fillna(fill_value)
        return filled_data
    
    def get_cleaning_report(self):
        report = {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'missing_values': self.data.isnull().sum().sum(),
            'numeric_columns': len(self.data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(self.data.select_dtypes(include=['object']).columns)
        }
        return report

def create_sample_dataset():
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'temperature': np.random.normal(25, 5, 100),
        'humidity': np.random.uniform(30, 90, 100),
        'pressure': np.random.normal(1013, 10, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    data.loc[np.random.choice(100, 5), 'temperature'] = np.nan
    data.loc[np.random.choice(100, 3), 'pressure'] = np.nan
    return data

if __name__ == "__main__":
    sample_data = create_sample_dataset()
    cleaner = DataCleaner(sample_data)
    
    print("Data Cleaning Utility")
    print("=" * 50)
    report = cleaner.get_cleaning_report()
    for key, value in report.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    cleaned_data = cleaner.handle_missing_values(strategy='mean')
    normalized_data = cleaner.normalize_minmax(['temperature', 'humidity', 'pressure'])
    
    print(f"\nOriginal data shape: {sample_data.shape}")
    print(f"Cleaned data shape: {cleaned_data.shape}")
    print(f"Normalized data sample:\n{normalized_data[['temperature', 'humidity', 'pressure']].head()}")
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
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns_to_clean = numeric_cols
    
    cleaned_df = df.copy()
    
    for column in columns_to_clean:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df
import pandas as pd
import sys

def remove_duplicates(input_file, output_file, subset=None):
    """
    Reads a CSV file, removes duplicate rows, and saves the cleaned data.
    If 'subset' is provided, only consider those columns for identifying duplicates.
    """
    try:
        df = pd.read_csv(input_file)
        initial_count = len(df)
        df_cleaned = df.drop_duplicates(subset=subset, keep='first')
        final_count = len(df_cleaned)
        df_cleaned.to_csv(output_file, index=False)
        print(f"Successfully removed {initial_count - final_count} duplicate rows.")
        print(f"Cleaned data saved to '{output_file}'.")
        return df_cleaned
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{input_file}' is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python data_cleaner.py <input_csv> <output_csv> [subset_columns]")
        print("Example: python data_cleaner.py raw_data.csv cleaned_data.csv")
        print("Example with subset: python data_cleaner.py raw_data.csv cleaned_data.csv 'col1,col2'")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    subset_cols = None
    if len(sys.argv) == 4:
        subset_cols = sys.argv[3].split(',')
        subset_cols = [col.strip() for col in subset_cols]

    remove_duplicates(input_csv, output_csv, subset_cols)