
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df

def z_score_normalize(dataframe, columns):
    """
    Apply z-score normalization to specified columns
    """
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col not in normalized_df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
        
        mean_val = normalized_df[col].mean()
        std_val = normalized_df[col].std()
        
        if std_val > 0:
            normalized_df[col] = (normalized_df[col] - mean_val) / std_val
        else:
            normalized_df[col] = 0
    
    return normalized_df

def min_max_normalize(dataframe, columns, feature_range=(0, 1)):
    """
    Apply min-max normalization to specified columns
    """
    normalized_df = dataframe.copy()
    min_val, max_val = feature_range
    
    for col in columns:
        if col not in normalized_df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
        
        col_min = normalized_df[col].min()
        col_max = normalized_df[col].max()
        
        if col_max > col_min:
            normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
            normalized_df[col] = normalized_df[col] * (max_val - min_val) + min_val
        else:
            normalized_df[col] = min_val
    
    return normalized_df

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy
    """
    processed_df = dataframe.copy()
    
    if columns is None:
        columns = processed_df.columns
    
    for col in columns:
        if col not in processed_df.columns:
            continue
            
        if processed_df[col].isnull().any():
            if strategy == 'mean':
                fill_value = processed_df[col].mean()
            elif strategy == 'median':
                fill_value = processed_df[col].median()
            elif strategy == 'mode':
                fill_value = processed_df[col].mode()[0]
            elif strategy == 'drop':
                processed_df = processed_df.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            processed_df[col] = processed_df[col].fillna(fill_value)
    
    return processed_df

def create_data_summary(dataframe):
    """
    Create comprehensive data summary
    """
    summary = {
        'total_rows': len(dataframe),
        'total_columns': len(dataframe.columns),
        'missing_values': dataframe.isnull().sum().sum(),
        'duplicate_rows': dataframe.duplicated().sum(),
        'numeric_columns': list(dataframe.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(dataframe.select_dtypes(include=['object']).columns),
        'column_types': dataframe.dtypes.to_dict()
    }
    
    return summarydef remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    Returns a new list with unique elements.
    """
    seen = set()
    unique_list = []
    for item in data_list:
        if item not in seen:
            seen.add(item)
            unique_list.append(item)
    return unique_list

def clean_numeric_data(values, default=0):
    """
    Convert all items in a list to integers, replacing non-convertible items with a default value.
    """
    cleaned = []
    for value in values:
        try:
            cleaned.append(int(value))
        except (ValueError, TypeError):
            cleaned.append(default)
    return cleaned

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 5]
    print("Original:", sample_data)
    print("Cleaned:", remove_duplicates(sample_data))
    
    numeric_mix = [10, "20", "abc", 30.5, None]
    print("Numeric original:", numeric_mix)
    print("Numeric cleaned:", clean_numeric_data(numeric_mix))import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    subset (list, optional): Column labels to consider for duplicates.
    keep (str, optional): Which duplicates to keep.
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def clean_numeric_column(df, column_name, fill_method='mean'):
    """
    Clean a numeric column by handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column_name (str): Name of the column to clean.
    fill_method (str): Method to fill missing values ('mean', 'median', 'zero').
    
    Returns:
    pd.DataFrame: DataFrame with cleaned column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df_clean = df.copy()
    
    if fill_method == 'mean':
        fill_value = df_clean[column_name].mean()
    elif fill_method == 'median':
        fill_value = df_clean[column_name].median()
    elif fill_method == 'zero':
        fill_value = 0
    else:
        raise ValueError("fill_method must be 'mean', 'median', or 'zero'")
    
    df_clean[column_name] = df_clean[column_name].fillna(fill_value)
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list, optional): List of required column names.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"
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
    
    return filtered_df

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
    Args:
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
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of column names to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[column]):
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Warning: Could not clean column '{column}': {e}")
    
    return cleaned_df
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, column, threshold=1.5):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        return self
        
    def remove_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[column]))
        self.df = self.df[z_scores < threshold]
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
        
    def fill_missing(self, column, method='mean'):
        if method == 'mean':
            fill_value = self.df[column].mean()
        elif method == 'median':
            fill_value = self.df[column].median()
        elif method == 'mode':
            fill_value = self.df[column].mode()[0]
        else:
            fill_value = method
            
        self.df[column].fillna(fill_value, inplace=True)
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
            'columns': list(self.df.columns)
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 1, 1000)
    }
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 50), 'feature_a'] = np.nan
    df.loc[np.random.choice(df.index, 30), 'feature_b'] = 500
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    cleaner.remove_outliers_iqr('feature_b', threshold=2.0)
    cleaner.fill_missing('feature_a', method='median')
    cleaner.normalize_column('feature_c', method='minmax')
    
    cleaned_df = cleaner.get_cleaned_data()
    summary = cleaner.get_summary()
    
    print(f"Data cleaning summary:")
    print(f"Original rows: {summary['original_rows']}")
    print(f"Cleaned rows: {summary['cleaned_rows']}")
    print(f"Rows removed: {summary['rows_removed']}")
    print(f"Columns: {summary['columns']}")
    print(f"\nCleaned data shape: {cleaned_df.shape}")
    print(f"Cleaned data head:\n{cleaned_df.head()}")
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
    
    if min_val == max_val:
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

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column not in cleaned_df.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, column)
        
        if normalize_method == 'minmax':
            cleaned_df[f'{column}_normalized'] = normalize_minmax(cleaned_df, column)
        elif normalize_method == 'zscore':
            cleaned_df[f'{column}_standardized'] = normalize_zscore(cleaned_df, column)
    
    return cleaned_df

def validate_data(df, required_columns=None, allow_nan=False):
    """
    Validate dataset structure and content
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if not allow_nan:
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            print(f"Warning: Dataset contains {nan_count} NaN values")
    
    return True
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from the specified column.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_summary_statistics(data, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing count, mean, std, min, and max.
    """
    stats = {
        'count': data[column].count(),
        'mean': data[column].mean(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max()
    }
    return statsimport pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fill_missing (bool): Whether to fill missing values.
        fill_value: Value to use for filling missing data.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else '', inplace=True)
            else:
                cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 0, inplace=True)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate the DataFrame for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names. Default is None.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    if df.empty:
        print("DataFrame is empty.")
        return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['A', 'B'])
    print(f"\nData validation result: {is_valid}")