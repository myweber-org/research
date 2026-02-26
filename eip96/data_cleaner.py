import pandas as pd

def clean_data(df):
    """
    Remove duplicate rows and fill missing values with column mean.
    """
    # Remove duplicates
    df_cleaned = df.drop_duplicates()
    
    # Fill missing numeric values with column mean
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
    
    # Fill missing categorical values with mode
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_cleaned[col].isnull().any():
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
    
    return df_cleaned

def main():
    # Example usage
    data = {
        'id': [1, 2, 2, 3, 4],
        'value': [10.5, None, 15.0, 20.0, None],
        'category': ['A', 'B', None, 'A', 'C']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_data(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)

if __name__ == "__main__":
    main()
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_basic_stats(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std()
    }
    return stats

if __name__ == "__main__":
    import pandas as pd
    
    # Example usage
    sample_data = pd.DataFrame({
        'values': np.random.normal(100, 15, 1000)
    })
    
    print("Original data shape:", sample_data.shape)
    cleaned_data = remove_outliers_iqr(sample_data, 'values')
    print("Cleaned data shape:", cleaned_data.shape)
    
    stats = calculate_basic_stats(cleaned_data, 'values')
    print("Statistics after cleaning:", stats)
import pandas as pd
import numpy as np

def clean_dataset(df, column_mapping=None, drop_na_threshold=0.5):
    """
    Clean a dataset by handling missing values, removing duplicates,
    and standardizing column names.
    """
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Standardize column names if mapping is provided
    if column_mapping:
        df_clean = df_clean.rename(columns=column_mapping)
    
    # Convert column names to lowercase and replace spaces with underscores
    df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
    
    # Remove duplicate rows
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)
    
    # Calculate missing value percentage for each column
    missing_percentage = df_clean.isnull().sum() / len(df_clean)
    
    # Drop columns with too many missing values
    columns_to_drop = missing_percentage[missing_percentage > drop_na_threshold].index
    df_clean = df_clean.drop(columns=columns_to_drop)
    
    # Fill missing values for numeric columns with median
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Fill missing values for categorical columns with mode
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    # Remove outliers using IQR method for numeric columns
    for col in numeric_cols:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them
            df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
            df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
    
    # Generate cleaning report
    report = {
        'original_rows': len(df),
        'cleaned_rows': len(df_clean),
        'duplicates_removed': duplicates_removed,
        'columns_dropped': list(columns_to_drop),
        'columns_remaining': list(df_clean.columns),
        'missing_values_before': df.isnull().sum().sum(),
        'missing_values_after': df_clean.isnull().sum().sum()
    }
    
    return df_clean, report

def validate_dataframe(df, required_columns=None, min_rows=10):
    """
    Validate that a dataframe meets basic requirements.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if len(df) < min_rows:
        raise ValueError(f"DataFrame must have at least {min_rows} rows")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'ID': [1, 2, 3, 4, 5, 5, 6, 7, 8, 9],
        'Name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Eve', 'Frank', 'Grace', None, 'Ivy'],
        'Age': [25, 30, 35, 40, 45, 45, 50, 55, 60, 200],
        'Salary': [50000, 60000, None, 80000, 90000, 90000, 100000, 110000, 120000, 130000],
        'Department': ['HR', 'IT', 'IT', 'Finance', 'Finance', 'Finance', 'HR', 'IT', None, 'IT']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df, report = clean_dataset(df)
    
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    print("Cleaning Report:")
    for key, value in report.items():
        print(f"{key}: {value}")
    
    # Validate the cleaned data
    try:
        validate_dataframe(cleaned_df, min_rows=5)
        print("\nData validation passed!")
    except ValueError as e:
        print(f"\nData validation failed: {e}")import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns):
    """
    Remove outliers using the Interquartile Range method.
    """
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def remove_outliers_zscore(df, columns, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            df_clean = df_clean[(z_scores < threshold) | (df[col].isna())]
    return df_clean

def normalize_minmax(df, columns):
    """
    Normalize specified columns using Min-Max scaling.
    """
    df_norm = df.copy()
    for col in columns:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df_norm[col] = 0
    return df_norm

def normalize_zscore(df, columns):
    """
    Normalize specified columns using Z-score standardization.
    """
    df_norm = df.copy()
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val != 0:
                df_norm[col] = (df[col] - mean_val) / std_val
            else:
                df_norm[col] = 0
    return df_norm

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    """
    df_filled = df.copy()
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if col in df.columns and df[col].isna().any():
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            elif strategy == 'drop':
                df_filled = df_filled.dropna(subset=[col])
                continue
            else:
                fill_value = 0
            
            df_filled[col] = df_filled[col].fillna(fill_value)
    
    return df_filled

def get_data_summary(df):
    """
    Generate a summary statistics dataframe.
    """
    summary = pd.DataFrame({
        'mean': df.mean(),
        'median': df.median(),
        'std': df.std(),
        'min': df.min(),
        'max': df.max(),
        'missing': df.isna().sum(),
        'missing_pct': (df.isna().sum() / len(df)) * 100
    })
    return summary