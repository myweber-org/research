
import numpy as np
import pandas as pd

def remove_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

def normalize_minmax(dataframe, column):
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    if max_val == min_val:
        return dataframe[column].apply(lambda x: 0.0)
    return dataframe[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def standardize_zscore(dataframe, column):
    mean_val = dataframe[column].mean()
    std_val = dataframe[column].std()
    if std_val == 0:
        return dataframe[column].apply(lambda x: 0.0)
    return dataframe[column].apply(lambda x: (x - mean_val) / std_val)

def clean_dataset(dataframe, numeric_columns):
    cleaned_df = dataframe.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df

def validate_dataframe(dataframe, required_columns):
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return Trueimport pandas as pd
import numpy as np

def clean_dataframe(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values.
        Options: 'mean', 'median', 'drop', 'fill_zero'.
    outlier_threshold (float): Number of standard deviations for outlier detection.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    if missing_strategy == 'mean':
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(
            cleaned_df[numeric_cols].mean()
        )
    elif missing_strategy == 'median':
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(
            cleaned_df[numeric_cols].median()
        )
    elif missing_strategy == 'fill_zero':
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(0)
    elif missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna(subset=numeric_cols)
    
    # Remove outliers using z-score method
    if outlier_threshold > 0:
        z_scores = np.abs((cleaned_df[numeric_cols] - cleaned_df[numeric_cols].mean()) / 
                         cleaned_df[numeric_cols].std())
        outlier_mask = (z_scores < outlier_threshold).all(axis=1)
        cleaned_df = cleaned_df[outlier_mask].reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize specified columns in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): Columns to normalize. If None, normalize all numeric columns.
    method (str): Normalization method. Options: 'minmax', 'zscore'.
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns.
    """
    normalized_df = df.copy()
    
    if columns is None:
        columns = normalized_df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in normalized_df.columns and pd.api.types.is_numeric_dtype(normalized_df[col]):
            if method == 'minmax':
                col_min = normalized_df[col].min()
                col_max = normalized_df[col].max()
                if col_max > col_min:
                    normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
            elif method == 'zscore':
                col_mean = normalized_df[col].mean()
                col_std = normalized_df[col].std()
                if col_std > 0:
                    normalized_df[col] = (normalized_df[col] - col_mean) / col_std
    
    return normalized_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataframe(df, missing_strategy='mean', outlier_threshold=2)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {message}")
    
    normalized = normalize_data(cleaned, method='minmax')
    print("\nNormalized DataFrame:")
    print(normalized)import pandas as pd
import numpy as np

def load_data(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers(df, column, threshold=3):
    """Remove outliers using Z-score method."""
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    return df[z_scores < threshold]

def normalize_column(df, column):
    """Normalize column values to range [0,1]."""
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_data(input_file, output_file):
    """Main data cleaning pipeline."""
    df = load_data(input_file)
    
    if 'value' in df.columns:
        df = remove_outliers(df, 'value')
        df = normalize_column(df, 'value')
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    return df

if __name__ == "__main__":
    cleaned_df = clean_data('raw_data.csv', 'cleaned_data.csv')
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
import pandas as pd
import numpy as np

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, fill_na=True):
    """
    Clean a pandas DataFrame by standardizing columns, removing duplicates,
    and handling missing values.
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_na:
        for col in cleaned_df.select_dtypes(include=[np.number]).columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].fillna('Unknown')
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def standardize_string_columns(df, columns):
    """
    Standardize string columns by converting to lowercase and stripping whitespace.
    """
    df_copy = df.copy()
    
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype(str).str.lower().str.strip()
    
    return df_copy

def validate_dataframe(df, required_columns=None, unique_columns=None):
    """
    Validate DataFrame structure and constraints.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if unique_columns:
        for col in unique_columns:
            if col in df.columns:
                if df[col].duplicated().any():
                    raise ValueError(f"Column '{col}' contains duplicate values")
    
    return True

def process_data_file(file_path, output_path=None, **clean_kwargs):
    """
    Load, clean, and optionally save a data file.
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .xlsx")
    
    cleaned_df = clean_dataframe(df, **clean_kwargs)
    
    if output_path:
        if output_path.endswith('.csv'):
            cleaned_df.to_csv(output_path, index=False)
        elif output_path.endswith('.xlsx'):
            cleaned_df.to_excel(output_path, index=False)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Alice', 'Charlie', None],
        'Age': [25, 30, 25, 35, np.nan],
        'City': ['New York', 'Los Angeles', 'new york', 'Chicago', 'Boston']
    })
    
    print("Original Data:")
    print(sample_data)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataframe(sample_data)
    cleaned = standardize_string_columns(cleaned, ['City'])
    
    print("Cleaned Data:")
    print(cleaned)
    
    try:
        validate_dataframe(cleaned, required_columns=['Name', 'Age'])
        print("\nData validation passed!")
    except ValueError as e:
        print(f"\nData validation failed: {e}")