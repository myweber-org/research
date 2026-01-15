
import pandas as pd
import numpy as np
from typing import List, Optional

def clean_dataframe(df: pd.DataFrame, 
                    drop_duplicates: bool = True,
                    fill_missing: Optional[str] = None,
                    columns_to_standardize: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Clean a pandas DataFrame by removing duplicates, handling missing values,
    and standardizing specified columns.
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    if fill_missing is not None:
        if fill_missing == 'mean':
            df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
        elif fill_missing == 'median':
            df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
        elif fill_missing == 'mode':
            df_clean = df_clean.fillna(df_clean.mode().iloc[0])
        elif fill_missing == 'zero':
            df_clean = df_clean.fillna(0)
    
    if columns_to_standardize:
        for col in columns_to_standardize:
            if col in df_clean.columns and df_clean[col].dtype in ['float64', 'int64']:
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                if std > 0:
                    df_clean[col] = (df_clean[col] - mean) / std
    
    return df_clean

def validate_dataframe(df: pd.DataFrame, 
                       required_columns: List[str]) -> bool:
    """
    Validate that a DataFrame contains all required columns and has no NaN values
    in those columns.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False
    
    nan_counts = df[required_columns].isna().sum()
    if nan_counts.any():
        print(f"NaN values found in required columns:\n{nan_counts[nan_counts > 0]}")
        return False
    
    return True

def sample_dataframe(df: pd.DataFrame, 
                     sample_size: int = 1000,
                     random_state: int = 42) -> pd.DataFrame:
    """
    Create a random sample from a DataFrame while maintaining relative distributions.
    """
    if len(df) <= sample_size:
        return df
    
    return df.sample(n=sample_size, random_state=random_state)

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'id': [1, 2, 3, 3, 4, 5],
        'value': [10.5, 20.3, np.nan, 30.1, 40.7, 50.2],
        'category': ['A', 'B', 'A', 'A', 'B', 'C']
    })
    
    print("Original data:")
    print(sample_data)
    
    cleaned_data = clean_dataframe(
        sample_data,
        drop_duplicates=True,
        fill_missing='mean',
        columns_to_standardize=['value']
    )
    
    print("\nCleaned data:")
    print(cleaned_data)
    
    is_valid = validate_dataframe(cleaned_data, ['id', 'value'])
    print(f"\nData validation: {is_valid}")
    
    sampled_data = sample_dataframe(cleaned_data, sample_size=3)
    print("\nSampled data:")
    print(sampled_data)
import pandas as pd

def clean_data(df, drop_duplicates=True, fill_missing=True):
    """
    Clean the input DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (bool): Whether to fill missing values. Default is True.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = cleaned_df.shape[0]
        cleaned_df = cleaned_df.drop_duplicates()
        removed_rows = initial_rows - cleaned_df.shape[0]
        print(f"Removed {removed_rows} duplicate rows.")
    
    if fill_missing:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
        
        categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
        cleaned_df[categorical_cols] = cleaned_df[categorical_cols].fillna('Unknown')
        
        print("Filled missing values in numeric and categorical columns.")
    
    return cleaned_df

def validate_data(df, required_columns):
    """
    Validate that the DataFrame contains all required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        bool: True if all required columns are present, False otherwise.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False
    else:
        print("All required columns are present.")
        return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 3, 4, None],
        'name': ['Alice', 'Bob', 'Charlie', 'Charlie', None, 'Eve'],
        'age': [25, 30, None, 35, 40, 45],
        'score': [85.5, 92.0, 78.5, 78.5, None, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned_df = clean_data(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    required_cols = ['id', 'name', 'age']
    is_valid = validate_data(cleaned_df, required_cols)
    print(f"\nData validation result: {is_valid}")
import pandas as pd
import numpy as np
from typing import List, Optional

def remove_duplicates(
    data: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first',
    inplace: bool = False
) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame with configurable parameters.
    
    Args:
        data: Input DataFrame
        subset: Column labels to consider for identifying duplicates
        keep: Which duplicates to keep ('first', 'last', False)
        inplace: Whether to modify the DataFrame in place
    
    Returns:
        DataFrame with duplicates removed
    """
    if not inplace:
        data = data.copy()
    
    if subset is None:
        subset = data.columns.tolist()
    
    cleaned_data = data.drop_duplicates(subset=subset, keep=keep, inplace=False)
    
    if inplace:
        data.drop_duplicates(subset=subset, keep=keep, inplace=True)
        return data
    
    return cleaned_data

def validate_dataframe(data: pd.DataFrame) -> bool:
    """
    Basic validation of DataFrame structure.
    
    Args:
        data: DataFrame to validate
    
    Returns:
        Boolean indicating if DataFrame is valid
    """
    if not isinstance(data, pd.DataFrame):
        return False
    
    if data.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if data.isnull().all().any():
        print("Warning: Some columns contain only null values")
    
    return True

def clean_numeric_columns(
    data: pd.DataFrame,
    columns: List[str],
    fill_value: float = 0.0
) -> pd.DataFrame:
    """
    Clean numeric columns by filling NaN values.
    
    Args:
        data: Input DataFrame
        columns: List of column names to clean
        fill_value: Value to use for filling NaN
    
    Returns:
        DataFrame with cleaned numeric columns
    """
    cleaned_data = data.copy()
    
    for col in columns:
        if col in cleaned_data.columns:
            cleaned_data[col] = pd.to_numeric(
                cleaned_data[col], 
                errors='coerce'
            ).fillna(fill_value)
    
    return cleaned_data

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'id': [1, 2, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David'],
        'score': [85, 90, 90, 78, np.nan],
        'age': [25, 30, 30, 35, 40]
    })
    
    print("Original data:")
    print(sample_data)
    print("\nAfter removing duplicates:")
    cleaned = remove_duplicates(sample_data, subset=['id', 'name'])
    print(cleaned)
    
    print("\nAfter cleaning numeric columns:")
    numeric_cleaned = clean_numeric_columns(cleaned, columns=['score'])
    print(numeric_cleaned)
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    mask = z_scores < threshold
    return data[mask]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].copy()
    
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
        return data[column].copy()
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Main function to clean dataset by removing outliers and normalizing numeric columns
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in df.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize_method == 'minmax':
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df[col] = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

def get_summary_statistics(df):
    """
    Generate summary statistics for numeric columns
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    summary = pd.DataFrame({
        'mean': df[numeric_cols].mean(),
        'median': df[numeric_cols].median(),
        'std': df[numeric_cols].std(),
        'min': df[numeric_cols].min(),
        'max': df[numeric_cols].max(),
        'count': df[numeric_cols].count(),
        'missing': df[numeric_cols].isnull().sum()
    })
    
    return summaryimport pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
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

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[column]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

def save_cleaned_data(df, input_path, suffix='_cleaned'):
    """
    Save cleaned DataFrame to a new CSV file.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame
        input_path (str): Original file path
        suffix (str): Suffix to add to filename
    
    Returns:
        str: Path to saved file
    """
    if not input_path.endswith('.csv'):
        raise ValueError("Input file must be a CSV file")
    
    output_path = input_path.replace('.csv', f'{suffix}.csv')
    df.to_csv(output_path, index=False)
    
    return output_path

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': range(1, 101),
        'value': np.concatenate([
            np.random.normal(100, 10, 90),
            np.random.normal(300, 50, 10)  # Outliers
        ]),
        'score': np.concatenate([
            np.random.normal(50, 5, 95),
            np.random.normal(200, 30, 5)  # Outliers
        ])
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Original data shape: {df.shape}")
    
    cleaned_df = clean_numeric_data(df, columns=['value', 'score'])
    print(f"Cleaned data shape: {cleaned_df.shape}")
    
    # Save to file (commented out for example)
    # output_file = save_cleaned_data(cleaned_df, 'input_data.csv')
    # print(f"Cleaned data saved to: {output_file}")