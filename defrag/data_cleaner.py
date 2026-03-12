
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers using IQR method
    """
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

def remove_outliers_zscore(dataframe, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    z_scores = np.abs(stats.zscore(dataframe[column]))
    return dataframe[z_scores < threshold]

def normalize_minmax(dataframe, column):
    """
    Normalize column using Min-Max scaling
    """
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    
    if max_val == min_val:
        return dataframe[column].apply(lambda x: 0.5)
    
    return (dataframe[column] - min_val) / (max_val - min_val)

def normalize_zscore(dataframe, column):
    """
    Normalize column using Z-score standardization
    """
    mean_val = dataframe[column].mean()
    std_val = dataframe[column].std()
    
    if std_val == 0:
        return dataframe[column].apply(lambda x: 0)
    
    return (dataframe[column] - mean_val) / std_val

def clean_dataset(dataframe, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Main cleaning function for numeric columns
    """
    cleaned_df = dataframe.copy()
    
    for column in numeric_columns:
        if column not in cleaned_df.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, column)
        
        if normalize_method == 'minmax':
            cleaned_df[column] = normalize_minmax(cleaned_df, column)
        elif normalize_method == 'zscore':
            cleaned_df[column] = normalize_zscore(cleaned_df, column)
    
    return cleaned_df.reset_index(drop=True)

def get_summary_statistics(dataframe, numeric_columns):
    """
    Generate summary statistics for numeric columns
    """
    summary = {}
    
    for column in numeric_columns:
        if column in dataframe.columns:
            col_data = dataframe[column]
            summary[column] = {
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'count': col_data.count(),
                'missing': col_data.isnull().sum()
            }
    
    return pd.DataFrame(summary).T

def detect_missing_patterns(dataframe, threshold=0.3):
    """
    Detect columns with high percentage of missing values
    """
    missing_percent = dataframe.isnull().sum() / len(dataframe)
    high_missing = missing_percent[missing_percent > threshold].index.tolist()
    
    return {
        'missing_percentage': missing_percent,
        'high_missing_columns': high_missing,
        'total_missing': dataframe.isnull().sum().sum()
    }import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean', drop_threshold=0.5):
    """
    Load and clean a CSV file by handling missing values.
    
    Parameters:
    filepath (str): Path to the CSV file.
    fill_strategy (str): Strategy for filling missing values.
                         Options: 'mean', 'median', 'mode', 'zero'.
    drop_threshold (float): Drop columns with missing ratio above this threshold.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at {filepath}")
    
    missing_ratio = df.isnull().sum() / len(df)
    columns_to_drop = missing_ratio[missing_ratio > drop_threshold].index
    df = df.drop(columns=columns_to_drop)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    if fill_strategy == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif fill_strategy == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif fill_strategy == 'zero':
        df[numeric_cols] = df[numeric_cols].fillna(0)
    elif fill_strategy == 'mode':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
    
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    return df

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to a new CSV file.
    
    Parameters:
    df (pd.DataFrame): Cleaned DataFrame.
    output_path (str): Path to save the cleaned CSV.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, fill_strategy='median', drop_threshold=0.3)
    save_cleaned_data(cleaned_df, output_file)
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
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
    Calculate summary statistics for a DataFrame column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing count, mean, std, min, max, and quartiles.
    """
    stats = {
        'count': data[column].count(),
        'mean': data[column].mean(),
        'std': data[column].std(),
        'min': data[column].min(),
        '25%': data[column].quantile(0.25),
        '50%': data[column].quantile(0.50),
        '75%': data[column].quantile(0.75),
        'max': data[column].max()
    }
    return stats