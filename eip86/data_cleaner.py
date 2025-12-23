
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_clean = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                df_clean = df_clean[mask]
        
        removed_count = self.original_shape[0] - df_clean.shape[0]
        self.df = df_clean.reset_index(drop=True)
        return removed_count
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val != min_val:
                    df_normalized[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        self.df = df_normalized
        return self
    
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_standardized = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    df_standardized[col] = (self.df[col] - mean_val) / std_val
        
        self.df = df_standardized
        return self
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_filled = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                median_val = self.df[col].median()
                df_filled[col] = self.df[col].fillna(median_val)
        
        self.df = df_filled
        return self
    
    def get_cleaned_data(self):
        return self.df.copy()
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'missing_values': self.df.isnull().sum().sum()
        }
        return summary

def example_usage():
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.uniform(0, 1, 1000)
    }
    
    df = pd.DataFrame(data)
    df.iloc[10:20, 0] = np.nan
    df.iloc[100:110, 1] = np.nan
    
    cleaner = DataCleaner(df)
    print(f"Initial shape: {cleaner.original_shape}")
    
    removed = cleaner.remove_outliers_iqr(['feature1', 'feature2'])
    print(f"Removed {removed} outliers")
    
    cleaner.fill_missing_median()
    cleaner.standardize_zscore(['feature1', 'feature2'])
    cleaner.normalize_minmax(['feature3'])
    
    cleaned_df = cleaner.get_cleaned_data()
    summary = cleaner.get_summary()
    
    print(f"Final shape: {cleaned_df.shape}")
    print(f"Summary: {summary}")
    
    return cleaned_df

if __name__ == "__main__":
    result_df = example_usage()
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range (IQR) method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from the specified column.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(file_path, columns_to_clean):
    """
    Load a dataset from a CSV file and remove outliers from specified columns.
    
    Parameters:
    file_path (str): Path to the CSV file.
    columns_to_clean (list): List of column names to clean.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Original dataset shape: {df.shape}")
        
        for column in columns_to_clean:
            if column in df.columns:
                original_count = len(df)
                df = remove_outliers_iqr(df, column)
                removed_count = original_count - len(df)
                print(f"Removed {removed_count} outliers from column '{column}'")
            else:
                print(f"Warning: Column '{column}' not found in dataset")
        
        print(f"Cleaned dataset shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Example usage
    cleaned_data = clean_dataset('sample_data.csv', ['age', 'salary', 'score'])
    
    if not cleaned_data.empty:
        cleaned_data.to_csv('cleaned_data.csv', index=False)
        print("Cleaned data saved to 'cleaned_data.csv'")
import pandas as pd
import numpy as np

def clean_csv_data(input_path, output_path, missing_strategy='mean'):
    """
    Load a CSV file, handle missing values, and save cleaned data.
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str): Path to save cleaned CSV file
        missing_strategy (str): Strategy for handling missing values.
            Options: 'mean', 'median', 'mode', 'drop'
    """
    try:
        df = pd.read_csv(input_path)
        print(f"Loaded data with shape: {df.shape}")
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values")
            
            if missing_strategy == 'drop':
                df_cleaned = df.dropna()
                print(f"Removed rows with missing values. New shape: {df_cleaned.shape}")
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    if df[col].isnull().any():
                        if missing_strategy == 'mean':
                            fill_value = df[col].mean()
                        elif missing_strategy == 'median':
                            fill_value = df[col].median()
                        elif missing_strategy == 'mode':
                            fill_value = df[col].mode()[0]
                        else:
                            fill_value = 0
                        
                        df[col].fillna(fill_value, inplace=True)
                        print(f"Filled missing values in '{col}' with {missing_strategy}: {fill_value}")
        else:
            df_cleaned = df
            print("No missing values found")
        
        df_cleaned.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return False
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return False

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Args:
        df (pd.DataFrame): Dataframe to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append("Dataframe is empty")
    
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        validation_results['warnings'].append(f"Found {duplicate_rows} duplicate rows")
    
    return validation_results

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [5, np.nan, 7, 8, 9],
        'C': ['x', 'y', 'z', 'x', 'y']
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    clean_csv_data('sample_data.csv', 'cleaned_sample.csv', missing_strategy='mean')
    
    import os
    if os.path.exists('sample_data.csv'):
        os.remove('sample_data.csv')
    if os.path.exists('cleaned_sample.csv'):
        os.remove('cleaned_sample.csv')
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='zscore'):
    """
    Comprehensive data cleaning pipeline
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column not in data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        
        if normalize_method == 'minmax':
            cleaned_data[column] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[column] = normalize_zscore(cleaned_data, column)
    
    return cleaned_data

def validate_data(data, required_columns=None, allow_nan=False):
    """
    Validate data structure and content
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if not allow_nan:
        nan_count = data.isnull().sum().sum()
        if nan_count > 0:
            raise ValueError(f"Data contains {nan_count} NaN values")
    
    return True