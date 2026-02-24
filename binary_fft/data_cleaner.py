
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df[col]))
        df = df[z_scores < 3]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = load_and_clean_data(input_file)
    save_cleaned_data(cleaned_df, output_file)
    print(f"Data cleaning complete. Cleaned data saved to {output_file}")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    return filtered_df

def normalize_minmax(dataframe, columns):
    """
    Normalize specified columns using min-max scaling
    """
    df_normalized = dataframe.copy()
    for col in columns:
        if col in df_normalized.columns:
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            if max_val != min_val:
                df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    return df_normalized

def standardize_zscore(dataframe, columns):
    """
    Standardize specified columns using z-score normalization
    """
    df_standardized = dataframe.copy()
    for col in columns:
        if col in df_standardized.columns:
            mean_val = df_standardized[col].mean()
            std_val = df_standardized[col].std()
            if std_val > 0:
                df_standardized[col] = (df_standardized[col] - mean_val) / std_val
    return df_standardized

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values with specified strategy
    """
    df_processed = dataframe.copy()
    
    if columns is None:
        columns = df_processed.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df_processed.columns and df_processed[col].isnull().any():
            if strategy == 'mean':
                fill_value = df_processed[col].mean()
            elif strategy == 'median':
                fill_value = df_processed[col].median()
            elif strategy == 'mode':
                fill_value = df_processed[col].mode()[0]
            elif strategy == 'drop':
                df_processed = df_processed.dropna(subset=[col])
                continue
            else:
                fill_value = 0
            
            df_processed[col] = df_processed[col].fillna(fill_value)
    
    return df_processed

def validate_dataframe(dataframe, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content
    """
    if dataframe.empty:
        raise ValueError("DataFrame is empty")
    
    if len(dataframe) < min_rows:
        raise ValueError(f"DataFrame has fewer than {min_rows} rows")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

def create_summary_statistics(dataframe):
    """
    Create comprehensive summary statistics
    """
    summary = {
        'row_count': len(dataframe),
        'column_count': len(dataframe.columns),
        'missing_values': dataframe.isnull().sum().to_dict(),
        'data_types': dataframe.dtypes.astype(str).to_dict(),
        'numeric_stats': {}
    }
    
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': dataframe[col].mean(),
            'median': dataframe[col].median(),
            'std': dataframe[col].std(),
            'min': dataframe[col].min(),
            'max': dataframe[col].max(),
            'q1': dataframe[col].quantile(0.25),
            'q3': dataframe[col].quantile(0.75)
        }
    
    return summary
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to process
    threshold (float): Multiplier for IQR (default 1.5)
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.DataFrame: Dataframe with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data
    
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to standardize
    
    Returns:
    pd.DataFrame: Dataframe with standardized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data
    
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(data, numeric_columns, outlier_threshold=1.5, normalization_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names to process
    outlier_threshold (float): Threshold for outlier removal
    normalization_method (str): 'minmax' or 'zscore' normalization
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, column, outlier_threshold)
            
            if normalization_method == 'minmax':
                cleaned_data = normalize_minmax(cleaned_data, column)
            elif normalization_method == 'zscore':
                cleaned_data = standardize_zscore(cleaned_data, column)
    
    return cleaned_data

def validate_data(data, required_columns):
    """
    Validate that required columns exist and have no null values.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    
    null_counts = data[required_columns].isnull().sum()
    if null_counts.any():
        print(f"Null values found:\n{null_counts[null_counts > 0]}")
        return False
    
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.uniform(0, 1, 1000)
    })
    
    print("Original data shape:", sample_data.shape)
    print("Original data statistics:")
    print(sample_data.describe())
    
    cleaned = clean_dataset(sample_data, ['feature1', 'feature2'], normalization_method='zscore')
    
    print("\nCleaned data shape:", cleaned.shape)
    print("Cleaned data statistics:")
    print(cleaned[['feature1_standardized', 'feature2_standardized']].describe())