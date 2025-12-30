
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    """
    Remove outliers using IQR method for specified columns.
    Returns cleaned DataFrame and outlier indices.
    """
    clean_df = df.copy()
    outlier_indices = []
    
    for col in columns:
        if col not in clean_df.columns:
            continue
            
        Q1 = clean_df[col].quantile(0.25)
        Q3 = clean_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        col_outliers = clean_df[(clean_df[col] < lower_bound) | (clean_df[col] > upper_bound)].index
        outlier_indices.extend(col_outliers)
        clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]
    
    return clean_df, list(set(outlier_indices))

def remove_outliers_zscore(df, columns, threshold=3):
    """
    Remove outliers using Z-score method.
    Returns cleaned DataFrame and outlier indices.
    """
    clean_df = df.copy()
    outlier_indices = []
    
    for col in columns:
        if col not in clean_df.columns:
            continue
            
        z_scores = np.abs(stats.zscore(clean_df[col].dropna()))
        col_outliers = clean_df[z_scores > threshold].index
        outlier_indices.extend(col_outliers)
        clean_df = clean_df[z_scores <= threshold]
    
    return clean_df, list(set(outlier_indices))

def normalize_minmax(df, columns):
    """
    Apply Min-Max normalization to specified columns.
    Returns DataFrame with normalized columns.
    """
    normalized_df = df.copy()
    
    for col in columns:
        if col not in normalized_df.columns:
            continue
            
        min_val = normalized_df[col].min()
        max_val = normalized_df[col].max()
        
        if max_val != min_val:
            normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
        else:
            normalized_df[col] = 0
    
    return normalized_df

def normalize_zscore(df, columns):
    """
    Apply Z-score normalization to specified columns.
    Returns DataFrame with normalized columns.
    """
    normalized_df = df.copy()
    
    for col in columns:
        if col not in normalized_df.columns:
            continue
            
        mean_val = normalized_df[col].mean()
        std_val = normalized_df[col].std()
        
        if std_val != 0:
            normalized_df[col] = (normalized_df[col] - mean_val) / std_val
        else:
            normalized_df[col] = 0
    
    return normalized_df

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    Supported strategies: 'mean', 'median', 'mode', 'drop'
    """
    if columns is None:
        columns = df.columns
    
    processed_df = df.copy()
    
    if strategy == 'drop':
        return processed_df.dropna(subset=columns)
    
    for col in columns:
        if col not in processed_df.columns:
            continue
            
        if strategy == 'mean':
            fill_value = processed_df[col].mean()
        elif strategy == 'median':
            fill_value = processed_df[col].median()
        elif strategy == 'mode':
            fill_value = processed_df[col].mode()[0] if not processed_df[col].mode().empty else 0
        else:
            fill_value = 0
        
        processed_df[col] = processed_df[col].fillna(fill_value)
    
    return processed_df

def clean_dataset(df, numerical_columns, categorical_columns=None, 
                  outlier_method='iqr', normalize_method='minmax',
                  missing_strategy='mean'):
    """
    Complete data cleaning pipeline.
    """
    if categorical_columns is None:
        categorical_columns = []
    
    all_columns = numerical_columns + categorical_columns
    
    print(f"Initial dataset shape: {df.shape}")
    
    cleaned_df = handle_missing_values(df, strategy=missing_strategy, columns=all_columns)
    print(f"After handling missing values: {cleaned_df.shape}")
    
    if outlier_method == 'iqr':
        cleaned_df, outliers = remove_outliers_iqr(cleaned_df, numerical_columns)
    elif outlier_method == 'zscore':
        cleaned_df, outliers = remove_outliers_zscore(cleaned_df, numerical_columns)
    else:
        outliers = []
    
    print(f"Outliers removed: {len(outliers)}")
    print(f"After outlier removal: {cleaned_df.shape}")
    
    if normalize_method == 'minmax':
        cleaned_df = normalize_minmax(cleaned_df, numerical_columns)
    elif normalize_method == 'zscore':
        cleaned_df = normalize_zscore(cleaned_df, numerical_columns)
    
    print(f"Final dataset shape: {cleaned_df.shape}")
    
    return cleaned_df, outliers

if __name__ == "__main__":
    sample_data = {
        'feature1': [1, 2, 3, 4, 100, 6, 7, 8, 9, 10],
        'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 1000],
        'feature3': [100, 200, 300, 400, 500, 600, 700, 800, 900, 10000],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    numerical_cols = ['feature1', 'feature2', 'feature3']
    categorical_cols = ['category']
    
    cleaned_data, removed_outliers = clean_dataset(
        df, 
        numerical_columns=numerical_cols,
        categorical_columns=categorical_cols,
        outlier_method='iqr',
        normalize_method='minmax',
        missing_strategy='mean'
    )
    
    print("\nCleaned Data Summary:")
    print(cleaned_data.describe())
import pandas as pd
import numpy as np

def remove_missing_rows(df, threshold=0.5):
    """
    Remove rows with missing values exceeding the threshold percentage.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        threshold (float): Maximum allowed missing value percentage per row (0-1)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    missing_percent = df.isnull().mean(axis=1)
    return df[missing_percent <= threshold].reset_index(drop=True)

def fill_missing_with_median(df, columns=None):
    """
    Fill missing values with column median for specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to process
    
    Returns:
        pd.DataFrame: DataFrame with filled values
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_filled = df.copy()
    for col in columns:
        if col in df.columns:
            median_val = df[col].median()
            df_filled[col] = df[col].fillna(median_val)
    
    return df_filled

def detect_outliers_iqr(df, column):
    """
    Detect outliers using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        pd.DataFrame: DataFrame containing outlier indices and values
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def cap_outliers(df, column, method='iqr'):
    """
    Cap outliers to specified bounds.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
        method (str): Method for determining bounds ('iqr' or 'percentile')
    
    Returns:
        pd.DataFrame: DataFrame with capped values
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_capped = df.copy()
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
    elif method == 'percentile':
        lower_bound = df[column].quantile(0.01)
        upper_bound = df[column].quantile(0.99)
    else:
        raise ValueError("Method must be 'iqr' or 'percentile'")
    
    df_capped[column] = df_capped[column].clip(lower=lower_bound, upper=upper_bound)
    return df_capped

def standardize_columns(df, columns=None):
    """
    Standardize specified columns using z-score normalization.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to standardize
    
    Returns:
        pd.DataFrame: DataFrame with standardized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_standardized = df.copy()
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df_standardized[col] = (df[col] - mean_val) / std_val
    
    return df_standardized

def clean_dataset(df, missing_threshold=0.3, outlier_columns=None):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        missing_threshold (float): Threshold for removing rows with missing values
        outlier_columns (list): Columns to process for outlier capping
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    cleaned_df = remove_missing_rows(cleaned_df, threshold=missing_threshold)
    cleaned_df = fill_missing_with_median(cleaned_df)
    
    if outlier_columns:
        for col in outlier_columns:
            if col in cleaned_df.columns:
                cleaned_df = cap_outliers(cleaned_df, col, method='iqr')
    
    return cleaned_df
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
                if max_val > min_val:
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
        return self.df
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_columns': self.df.shape[1],
            'removed_rows': self.original_shape[0] - self.df.shape[0],
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(exclude=[np.number]).columns)
        }
        return summary

def example_usage():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.randint(1, 100, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 50), 'feature_a'] = np.nan
    
    cleaner = DataCleaner(df)
    removed = cleaner.remove_outliers_iqr(['feature_a', 'feature_b'])
    cleaner.fill_missing_median()
    cleaner.standardize_zscore(['feature_a', 'feature_b'])
    
    cleaned_df = cleaner.get_cleaned_data()
    summary = cleaner.get_summary()
    
    print(f"Removed {removed} outliers")
    print(f"Final shape: {cleaned_df.shape}")
    print(f"Summary: {summary}")
    
    return cleaned_df

if __name__ == "__main__":
    result = example_usage()
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: Z-score threshold (default 3)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_indices = np.where(z_scores < threshold)[0]
    
    # Map indices back to original DataFrame indices
    valid_indices = data[column].dropna().index[filtered_indices]
    filtered_data = data.loc[valid_indices]
    
    return filtered_data.copy()

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling to range [0, 1].
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    normalized_data = data.copy()
    col_min = normalized_data[column].min()
    col_max = normalized_data[column].max()
    
    if col_max == col_min:
        normalized_data[column] = 0.5
    else:
        normalized_data[column] = (normalized_data[column] - col_min) / (col_max - col_min)
    
    return normalized_data

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with standardized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    standardized_data = data.copy()
    col_mean = standardized_data[column].mean()
    col_std = standardized_data[column].std()
    
    if col_std == 0:
        standardized_data[column] = 0
    else:
        standardized_data[column] = (standardized_data[column] - col_mean) / col_std
    
    return standardized_data

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric columns)
        outlier_method: 'iqr', 'zscore', or None
        normalize_method: 'minmax', 'zscore', or None
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_data = data.copy()
    
    if numeric_columns is None:
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove outliers
    if outlier_method == 'iqr':
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data = remove_outliers_iqr(cleaned_data, col)
    elif outlier_method == 'zscore':
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data = remove_outliers_zscore(cleaned_data, col)
    
    # Normalize data
    if normalize_method == 'minmax':
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data = normalize_minmax(cleaned_data, col)
    elif normalize_method == 'zscore':
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data = normalize_zscore(cleaned_data, col)
    
    return cleaned_data

def validate_data(data, required_columns=None, allow_nan=False, min_rows=1):
    """
    Validate dataset structure and content.
    
    Args:
        data: pandas DataFrame to validate
        required_columns: list of required column names
        allow_nan: whether NaN values are allowed
        min_rows: minimum number of rows required
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(data, pd.DataFrame):
        return False, "Input must be a pandas DataFrame"
    
    if len(data) < min_rows:
        return False, f"DataFrame must have at least {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if not allow_nan and data.isnull().any().any():
        return False, "DataFrame contains NaN values"
    
    return True, "Data validation passed"