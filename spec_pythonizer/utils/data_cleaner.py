
import numpy as np
import pandas as pd

def remove_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

def normalize_column(dataframe, column, method='minmax'):
    if method == 'minmax':
        min_val = dataframe[column].min()
        max_val = dataframe[column].max()
        if max_val != min_val:
            dataframe[column] = (dataframe[column] - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean_val = dataframe[column].mean()
        std_val = dataframe[column].std()
        if std_val > 0:
            dataframe[column] = (dataframe[column] - mean_val) / std_val
    return dataframe

def clean_dataset(dataframe, numeric_columns):
    cleaned_df = dataframe.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_column(cleaned_df, col, method='zscore')
    return cleaned_df.reset_index(drop=True)import pandas as pd
import numpy as np
from scipy import stats

def load_data(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    """Remove outliers using IQR method for a specific column."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    """Normalize a column using min-max scaling."""
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(input_file, output_file):
    """Main function to clean dataset."""
    df = load_data(input_file)
    
    print(f"Original dataset shape: {df.shape}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        df = remove_outliers_iqr(df, col)
    
    print(f"After outlier removal shape: {df.shape}")
    
    for col in numeric_cols:
        df = normalize_column(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned dataset saved to: {output_file}")
    
    return df

if __name__ == "__main__":
    cleaned_df = clean_dataset('raw_data.csv', 'cleaned_data.csv')
    print("Data cleaning completed successfully.")
import pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(filepath: str, 
                   missing_strategy: str = 'drop',
                   fill_value: Optional[float] = None) -> pd.DataFrame:
    """
    Load and clean CSV data by handling missing values.
    
    Args:
        filepath: Path to CSV file
        missing_strategy: 'drop', 'fill', or 'interpolate'
        fill_value: Value to use when missing_strategy is 'fill'
    
    Returns:
        Cleaned DataFrame
    """
    df = pd.read_csv(filepath)
    
    if missing_strategy == 'drop':
        df = df.dropna()
    elif missing_strategy == 'fill':
        if fill_value is not None:
            df = df.fillna(fill_value)
        else:
            df = df.fillna(df.mean(numeric_only=True))
    elif missing_strategy == 'interpolate':
        df = df.interpolate(method='linear', limit_direction='forward')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())
    
    return df.reset_index(drop=True)

def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate DataFrame has no infinite values and all numeric columns are finite."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        return not np.any(np.isinf(df[numeric_cols].values))
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, 4, 5],
        'C': [1, 2, 3, 4, 5]
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned = clean_csv_data('sample_data.csv', missing_strategy='fill')
    print(f"Data cleaned successfully: {validate_dataframe(cleaned)}")
    print(f"Shape: {cleaned.shape}")
    print(cleaned.head())