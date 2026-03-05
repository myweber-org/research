
import pandas as pd

def clean_dataset(df, drop_na=True, column_case='lower'):
    """
    Clean a pandas DataFrame by handling missing values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_na (bool): If True, drop rows with any null values. Default True.
    column_case (str): Target case for column names ('lower', 'upper', 'title'). Default 'lower'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if drop_na:
        cleaned_df = cleaned_df.dropna()
    else:
        # Fill numeric columns with median, object columns with mode
        for col in cleaned_df.columns:
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
    
    # Standardize column names
    if column_case == 'lower':
        cleaned_df.columns = cleaned_df.columns.str.lower()
    elif column_case == 'upper':
        cleaned_df.columns = cleaned_df.columns.str.upper()
    elif column_case == 'title':
        cleaned_df.columns = cleaned_df.columns.str.title()
    
    # Remove leading/trailing whitespace from column names
    cleaned_df.columns = cleaned_df.columns.str.strip()
    
    # Replace spaces with underscores in column names
    cleaned_df.columns = cleaned_df.columns.str.replace(' ', '_')
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    dict: Dictionary with validation results.
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check if input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        validation_result['is_valid'] = False
        validation_result['errors'].append('Input is not a pandas DataFrame')
        return validation_result
    
    # Check for empty DataFrame
    if df.empty:
        validation_result['warnings'].append('DataFrame is empty')
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f'Missing required columns: {missing_columns}')
    
    # Check for duplicate columns
    if len(df.columns) != len(set(df.columns)):
        validation_result['warnings'].append('Duplicate column names detected')
    
    return validation_result

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'Name': ['Alice', 'Bob', None, 'David'],
        'Age': [25, None, 30, 35],
        'City': ['New York', 'London', 'Paris', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned = clean_dataset(df, drop_na=False, column_case='lower')
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    # Validate the cleaned data
    validation = validate_dataframe(cleaned, required_columns=['name', 'age'])
    print("Validation Results:")
    print(f"Is valid: {validation['is_valid']}")
    print(f"Errors: {validation['errors']}")
    print(f"Warnings: {validation['warnings']}")import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_method='iqr', columns=None):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values.
        Options: 'mean', 'median', 'mode', 'drop'.
    outlier_method (str): Method for detecting outliers.
        Options: 'iqr', 'zscore'.
    columns (list): Specific columns to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if df_clean[col].dtype in [np.float64, np.int64]:
            handle_missing_values(df_clean, col, missing_strategy)
            handle_outliers(df_clean, col, outlier_method)
    
    return df_clean

def handle_missing_values(df, column, strategy):
    """Handle missing values in a specific column."""
    
    if strategy == 'mean':
        fill_value = df[column].mean()
    elif strategy == 'median':
        fill_value = df[column].median()
    elif strategy == 'mode':
        fill_value = df[column].mode()[0] if not df[column].mode().empty else 0
    elif strategy == 'drop':
        df.dropna(subset=[column], inplace=True)
        return
    else:
        raise ValueError(f"Unsupported missing value strategy: {strategy}")
    
    df[column].fillna(fill_value, inplace=True)

def handle_outliers(df, column, method):
    """Detect and cap outliers in a specific column."""
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
        
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        
        if std_val > 0:
            z_scores = np.abs((df[column] - mean_val) / std_val)
            df[column] = np.where(z_scores > 3, mean_val, df[column])
    
    else:
        raise ValueError(f"Unsupported outlier method: {method}")

def validate_cleaning(df_before, df_after):
    """Validate the cleaning process by comparing statistics."""
    
    print("Data Cleaning Validation Report")
    print("=" * 40)
    
    numeric_cols = df_before.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col in df_after.columns:
            print(f"\nColumn: {col}")
            print(f"  Missing values before: {df_before[col].isnull().sum()}")
            print(f"  Missing values after:  {df_after[col].isnull().sum()}")
            print(f"  Mean before: {df_before[col].mean():.2f}")
            print(f"  Mean after:  {df_after[col].mean():.2f}")
            print(f"  Std before:  {df_before[col].std():.2f}")
            print(f"  Std after:   {df_after[col].std():.2f}")

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, 20, 30, np.nan, 50, 200],
        'C': ['x', 'y', 'z', 'x', 'y', 'z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*40 + "\n")
    
    df_clean = clean_dataset(df, missing_strategy='median', outlier_method='iqr')
    print("Cleaned DataFrame:")
    print(df_clean)
    print("\n" + "="*40 + "\n")
    
    validate_cleaning(df, df_clean)