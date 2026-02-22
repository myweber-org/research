import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        remove_duplicates (bool): Whether to remove duplicate rows.
        fill_missing (str or dict): Strategy to fill missing values.
            Options: 'mean', 'median', 'mode', or a dictionary of column:value.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df.fillna(fill_missing, inplace=True)
        elif fill_missing == 'mean':
            cleaned_df.fillna(cleaned_df.mean(numeric_only=True), inplace=True)
        elif fill_missing == 'median':
            cleaned_df.fillna(cleaned_df.median(numeric_only=True), inplace=True)
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
    
    if remove_duplicates:
        cleaned_df.drop_duplicates(inplace=True)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of columns that must be present.
    
    Returns:
        dict: Validation results with keys 'is_valid' and 'issues'.
    """
    validation_result = {
        'is_valid': True,
        'issues': []
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"Missing columns: {missing_columns}")
    
    if df.empty:
        validation_result['is_valid'] = False
        validation_result['issues'].append("DataFrame is empty")
    
    null_counts = df.isnull().sum()
    if null_counts.any():
        validation_result['issues'].append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
    
    return validation_result

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, None, 4, 2],
        'B': ['x', 'y', 'z', None, 'x'],
        'C': [10.5, 20.3, 15.7, None, 10.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    validation = validate_data(cleaned, required_columns=['A', 'B', 'C'])
    print("\nValidation Result:")
    print(validation)
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
    """
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max(),
        'count': data[column].count()
    }
    return stats