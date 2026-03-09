
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
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. 
                        Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values.")
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[col].median()
                elif fill_missing == 'mode':
                    fill_value = cleaned_df[col].mode()[0]
                
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                print(f"Filled missing values in column '{col}' with {fill_missing}: {fill_value:.2f}")
    
    categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if cleaned_df[col].isnull().any():
            cleaned_df[col] = cleaned_df[col].fillna('Unknown')
            print(f"Filled missing values in categorical column '{col}' with 'Unknown'")
    
    print(f"Data cleaning complete. Final dataset shape: {cleaned_df.shape}")
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the structure and content of a DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame.")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    print("Data validation passed.")
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, np.nan, 15.7, np.nan],
        'category': ['A', 'B', 'B', 'C', None, 'A'],
        'score': [85, 92, 92, 78, 88, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='median')
    print("\nCleaned dataset:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['id', 'value', 'category'], min_rows=3)
    print(f"\nDataset validation result: {is_valid}")
import re
import pandas as pd
from typing import Optional, List, Dict, Any

def remove_special_characters(text: str, keep_chars: str = "") -> str:
    """
    Remove special characters from a string, optionally keeping specified characters.
    """
    if not isinstance(text, str):
        return text
    pattern = f"[^a-zA-Z0-9\s{re.escape(keep_chars)}]"
    return re.sub(pattern, '', text)

def validate_email(email: str) -> bool:
    """
    Validate an email address format.
    """
    if not isinstance(email, str):
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def normalize_phone_number(phone: str, country_code: str = "+1") -> Optional[str]:
    """
    Normalize a phone number to a standard format.
    """
    if not isinstance(phone, str):
        return None
    digits = re.sub(r'\D', '', phone)
    if len(digits) == 10:
        return f"{country_code}{digits}"
    elif len(digits) == 11 and digits.startswith('1'):
        return f"+{digits}"
    else:
        return None

def clean_dataframe(df: pd.DataFrame, text_columns: List[str] = None) -> pd.DataFrame:
    """
    Clean a DataFrame by removing special characters from specified text columns.
    """
    df_clean = df.copy()
    if text_columns is None:
        text_columns = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    for col in text_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(lambda x: remove_special_characters(str(x)) if pd.notna(x) else x)
    
    return df_clean

def check_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check for missing values in a DataFrame and return statistics.
    """
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    return {
        'missing_counts': missing_counts.to_dict(),
        'missing_percentages': missing_percentages.to_dict(),
        'total_missing': missing_counts.sum(),
        'columns_with_missing': missing_counts[missing_counts > 0].index.tolist()
    }