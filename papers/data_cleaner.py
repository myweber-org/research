import pandas as pd
import numpy as np
from scipy import stats

def load_data(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    """Remove outliers using IQR method for specified column."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def remove_outliers_zscore(df, column, threshold=3):
    """Remove outliers using Z-score method."""
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def normalize_minmax(df, column):
    """Normalize column using Min-Max scaling."""
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def normalize_zscore(df, column):
    """Normalize column using Z-score standardization."""
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def clean_dataset(input_file, output_file, outlier_method='iqr', normalize_method='minmax'):
    """Main function to clean dataset by removing outliers and normalizing."""
    df = load_data(input_file)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        if outlier_method == 'iqr':
            df = remove_outliers_iqr(df, col)
        elif outlier_method == 'zscore':
            df = remove_outliers_zscore(df, col)
    
    for col in numeric_cols:
        if normalize_method == 'minmax':
            df = normalize_minmax(df, col)
        elif normalize_method == 'zscore':
            df = normalize_zscore(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    return df

if __name__ == "__main__":
    cleaned_df = clean_dataset('raw_data.csv', 'cleaned_data.csv')
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
    
    return filtered_df.reset_index(drop=True)

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

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    
    data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100)
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("Original statistics:", calculate_summary_statistics(df, 'value'))
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned statistics:", calculate_summary_statistics(cleaned_df, 'value'))
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = example_usage()
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Args:
        dataframe: pandas DataFrame
        column: column name to process
        threshold: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df

def normalize_column_zscore(dataframe, column):
    """
    Normalize a column using z-score normalization.
    
    Args:
        dataframe: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column added as '{column}_normalized'
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    col_data = dataframe[column].values
    z_scores = stats.zscore(col_data, nan_policy='omit')
    
    normalized_col_name = f"{column}_normalized"
    dataframe[normalized_col_name] = z_scores
    
    return dataframe

def clean_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Args:
        dataframe: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'drop')
        columns: list of columns to process (None for all numeric columns)
    
    Returns:
        DataFrame with missing values handled
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    df_copy = dataframe.copy()
    
    for col in columns:
        if col not in df_copy.columns:
            continue
            
        if strategy == 'mean':
            fill_value = df_copy[col].mean()
        elif strategy == 'median':
            fill_value = df_copy[col].median()
        elif strategy == 'mode':
            fill_value = df_copy[col].mode()[0] if not df_copy[col].mode().empty else np.nan
        elif strategy == 'drop':
            df_copy = df_copy.dropna(subset=[col])
            continue
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        df_copy[col] = df_copy[col].fillna(fill_value)
    
    return df_copy

def validate_dataframe(dataframe, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe: pandas DataFrame to validate
        required_columns: list of required column names
        min_rows: minimum number of rows required
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(dataframe) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def create_sample_data():
    """
    Create sample data for testing the cleaning functions.
    
    Returns:
        pandas DataFrame with sample data
    """
    np.random.seed(42)
    
    data = {
        'id': range(1, 101),
        'value': np.random.normal(100, 15, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'score': np.random.uniform(0, 1, 100)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[np.random.choice(100, 5, replace=False), 'value'] = np.nan
    df.loc[np.random.choice(100, 3, replace=False), 'score'] = np.nan
    
    outlier_indices = np.random.choice(100, 3, replace=False)
    df.loc[outlier_indices, 'value'] = [250, -50, 300]
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Original DataFrame shape:", sample_df.shape)
    
    is_valid, message = validate_dataframe(sample_df, ['id', 'value', 'score'])
    print(f"Validation: {message}")
    
    cleaned_df = clean_missing_values(sample_df, strategy='mean')
    print("After missing value handling:", cleaned_df.shape)
    
    filtered_df = remove_outliers_iqr(cleaned_df, 'value')
    print("After outlier removal:", filtered_df.shape)
    
    normalized_df = normalize_column_zscore(filtered_df, 'value')
    print("Data cleaning completed successfully")
    print("Final columns:", normalized_df.columns.tolist())