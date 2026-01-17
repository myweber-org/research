
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataset(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        print(f"Original dataset shape: {df.shape}")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            original_count = len(df)
            df = remove_outliers_iqr(df, column)
            removed_count = original_count - len(df)
            if removed_count > 0:
                print(f"Removed {removed_count} outliers from column: {column}")
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned dataset saved to: {output_file}")
        print(f"Final dataset shape: {df.shape}")
        
        return True
    except Exception as e:
        print(f"Error during cleaning: {e}")
        return False

if __name__ == "__main__":
    input_path = "raw_data.csv"
    output_path = "cleaned_data.csv"
    clean_dataset(input_path, output_path)
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, standardize_columns=True):
    """
    Clean a pandas DataFrame by removing duplicates and standardizing column names.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        standardize_columns (bool): Whether to standardize column names.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if standardize_columns:
        cleaned_df.columns = (
            cleaned_df.columns
            .str.strip()
            .str.lower()
            .str.replace(' ', '_')
            .str.replace(r'[^\w_]', '', regex=True)
        )
        print("Column names standardized.")
    
    return cleaned_df

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in specified columns using different strategies.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        strategy (str): Strategy to use ('mean', 'median', 'mode', 'drop').
        columns (list): List of columns to process, None for all columns.
    
    Returns:
        pd.DataFrame: DataFrame with handled missing values.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    processed_df = df.copy()
    
    for col in columns:
        if col not in processed_df.columns:
            continue
            
        if strategy == 'mean':
            fill_value = processed_df[col].mean()
        elif strategy == 'median':
            fill_value = processed_df[col].median()
        elif strategy == 'mode':
            fill_value = processed_df[col].mode()[0] if not processed_df[col].mode().empty else np.nan
        elif strategy == 'drop':
            processed_df = processed_df.dropna(subset=[col])
            continue
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        missing_count = processed_df[col].isna().sum()
        if missing_count > 0:
            processed_df[col] = processed_df[col].fillna(fill_value)
            print(f"Filled {missing_count} missing values in '{col}' using {strategy}.")
    
    return processed_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame."
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows."
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid."

if __name__ == "__main__":
    sample_data = {
        'Customer ID': [1, 2, 2, 3, 4],
        'Order Date': ['2023-01-01', '2023-01-02', '2023-01-02', None, '2023-01-03'],
        'Product Name': ['Widget A', 'Widget B', 'Widget B', 'Widget C', None],
        'Price': [100.0, 200.0, 200.0, 150.0, 180.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataframe(df)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    validated, message = validate_dataframe(cleaned, required_columns=['customer_id', 'price'])
    print(f"\nValidation: {message}")
import pandas as pd
import numpy as np

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy to handle missing values ('mean', 'median', 'mode', 'drop')
    columns (list): List of columns to apply the strategy to. If None, applies to all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    df_copy = df.copy()
    
    if columns is None:
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    if strategy == 'drop':
        return df_copy.dropna(subset=columns)
    
    for col in columns:
        if col not in df_copy.columns:
            continue
            
        if strategy == 'mean':
            fill_value = df_copy[col].mean()
        elif strategy == 'median':
            fill_value = df_copy[col].median()
        elif strategy == 'mode':
            fill_value = df_copy[col].mode()[0] if not df_copy[col].mode().empty else np.nan
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        df_copy[col] = df_copy[col].fillna(fill_value)
    
    return df_copy

def remove_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Remove outliers using the Interquartile Range (IQR) method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of columns to check for outliers. If None, uses all numeric columns.
    threshold (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    df_copy = df.copy()
    
    if columns is None:
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    mask = pd.Series([True] * len(df_copy))
    
    for col in columns:
        if col not in df_copy.columns:
            continue
            
        Q1 = df_copy[col].quantile(0.25)
        Q3 = df_copy[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        col_mask = (df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)
        mask = mask & col_mask
    
    return df_copy[mask]

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize data in specified columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of columns to normalize. If None, uses all numeric columns.
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    df_copy = df.copy()
    
    if columns is None:
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col not in df_copy.columns:
            continue
            
        if method == 'minmax':
            min_val = df_copy[col].min()
            max_val = df_copy[col].max()
            if max_val != min_val:
                df_copy[col] = (df_copy[col] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean_val = df_copy[col].mean()
            std_val = df_copy[col].std()
            if std_val != 0:
                df_copy[col] = (df_copy[col] - mean_val) / std_val
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    return df_copy

def clean_dataset(df, missing_strategy='mean', outlier_threshold=1.5, normalize=False):
    """
    Complete data cleaning pipeline.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    missing_strategy (str): Strategy for handling missing values
    outlier_threshold (float): IQR threshold for outlier removal
    normalize (bool): Whether to normalize the data
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
        cleaned_df = remove_outliers_iqr(cleaned_df, threshold=outlier_threshold)
        
        if normalize:
            cleaned_df = normalize_data(cleaned_df, method='minmax')
    
    return cleaned_df