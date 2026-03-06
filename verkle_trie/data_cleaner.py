
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Column labels to consider for duplicates
        keep (str, optional): Which duplicates to keep: 'first', 'last', or False
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate row(s)")
    
    return cleaned_df

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns
    """
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

def get_data_summary(df):
    """
    Generate a summary of the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        dict: Summary statistics
    """
    summary = {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum()
    }
    
    return summary
import pandas as pd
import numpy as np
import re

def clean_column_names(df):
    df.columns = [re.sub(r'\s+', '_', col.strip().lower()) for col in df.columns]
    return df

def remove_duplicates(df, subset=None):
    return df.drop_duplicates(subset=subset, keep='first')

def handle_missing_values(df, strategy='mean', columns=None):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_copy = df.copy()
    for col in columns:
        if strategy == 'mean' and df_copy[col].dtype in [np.float64, np.int64]:
            df_copy[col].fillna(df_copy[col].mean(), inplace=True)
        elif strategy == 'median' and df_copy[col].dtype in [np.float64, np.int64]:
            df_copy[col].fillna(df_copy[col].median(), inplace=True)
        elif strategy == 'mode':
            df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
        elif strategy == 'drop':
            df_copy = df_copy.dropna(subset=[col])
        else:
            df_copy[col].fillna('', inplace=True)
    return df_copy

def normalize_text(df, columns):
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype(str).str.lower().str.strip()
    return df_copy

def validate_data(df, rules):
    errors = []
    for rule in rules:
        column, condition = rule
        if column in df.columns:
            invalid_rows = df[~condition(df[column])]
            if not invalid_rows.empty:
                errors.append(f"Rule violation in column '{column}': {len(invalid_rows)} rows")
    return errors

def process_csv(input_path, output_path, cleaning_steps=None):
    try:
        df = pd.read_csv(input_path)
        
        if cleaning_steps is None:
            cleaning_steps = [
                ('clean_column_names', {}),
                ('remove_duplicates', {'subset': None}),
                ('handle_missing_values', {'strategy': 'mean'}),
            ]
        
        for step_name, kwargs in cleaning_steps:
            if hasattr(__name__, step_name):
                func = globals()[step_name]
                df = func(df, **kwargs)
        
        df.to_csv(output_path, index=False)
        return True, f"Data cleaned successfully. Output saved to {output_path}"
    
    except Exception as e:
        return False, f"Error processing file: {str(e)}"

if __name__ == "__main__":
    result, message = process_csv('input.csv', 'cleaned_output.csv')
    print(message)
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path, numeric_columns):
    df = pd.read_csv(file_path)
    
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
            df = normalize_minmax(df, col)
    
    df = df.dropna()
    return df

def calculate_statistics(df, column):
    stats_dict = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'skewness': stats.skew(df[column].dropna()),
        'kurtosis': stats.kurtosis(df[column].dropna())
    }
    return stats_dict

if __name__ == "__main__":
    data = pd.DataFrame({
        'value': np.random.normal(100, 15, 1000),
        'score': np.random.uniform(0, 1, 1000)
    })
    
    cleaned_data = clean_dataset('sample_data.csv', ['value', 'score'])
    stats_info = calculate_statistics(cleaned_data, 'value')
    
    print(f"Original shape: {data.shape}")
    print(f"Cleaned shape: {cleaned_data.shape}")
    print(f"Statistics: {stats_info}")
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result