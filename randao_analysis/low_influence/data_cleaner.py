
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
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
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def example_usage():
    """
    Demonstrate the usage of the data cleaning functions.
    """
    np.random.seed(42)
    
    data = {
        'id': range(1, 101),
        'value': np.concatenate([
            np.random.normal(100, 15, 90),
            np.random.normal(300, 50, 10)
        ])
    }
    
    df = pd.DataFrame(data)
    
    print("Original data shape:", df.shape)
    print("Original statistics:", calculate_summary_statistics(df, 'value'))
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    
    print("\nCleaned data shape:", cleaned_df.shape)
    print("Cleaned statistics:", calculate_summary_statistics(cleaned_df, 'value'))
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = example_usage()
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_method=None):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Args:
        df: pandas DataFrame to clean
        drop_duplicates: Boolean, whether to drop duplicate rows
        fill_method: Method to fill missing values ('mean', 'median', 'mode', or None)
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        
        if fill_method == 'mean':
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(
                cleaned_df[numeric_cols].mean()
            )
        elif fill_method == 'median':
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(
                cleaned_df[numeric_cols].median()
            )
        elif fill_method == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    cleaned_df[col] = cleaned_df[col].fillna(
                        cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else ''
                    )
        else:
            raise ValueError(f"Unsupported fill method: {fill_method}")
    
    # Drop duplicates if requested
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of column names that must be present
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'column_types': df.dtypes.to_dict()
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['missing_required_columns'] = missing_columns
    
    return validation_results

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column using specified method.
    
    Args:
        df: pandas DataFrame
        column: Column name to process
        method: Outlier detection method ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    elif method == 'zscore':
        from scipy import stats
        z_scores = stats.zscore(df[column].dropna())
        abs_z_scores = abs(z_scores)
        filtered_df = df[abs_z_scores < threshold]
    
    else:
        raise ValueError(f"Unsupported outlier detection method: {method}")
    
    return filtered_dfimport csv
import re

def clean_numeric_string(value):
    """Remove non-numeric characters from a string and convert to float."""
    if not value or not isinstance(value, str):
        return None
    cleaned = re.sub(r'[^\d.-]', '', value)
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None

def standardize_phone_number(phone):
    """Standardize phone number format to (XXX) XXX-XXXX."""
    if not phone:
        return None
    digits = re.sub(r'\D', '', str(phone))
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    return phone

def read_csv_with_cleaning(filepath):
    """Read CSV file and apply basic cleaning transformations."""
    cleaned_data = []
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                cleaned_row = {}
                for key, value in row.items():
                    if 'phone' in key.lower():
                        cleaned_row[key] = standardize_phone_number(value)
                    elif any(num_key in key.lower() for num_key in ['price', 'amount', 'quantity']):
                        cleaned_row[key] = clean_numeric_string(value)
                    else:
                        cleaned_row[key] = value.strip() if isinstance(value, str) else value
                cleaned_data.append(cleaned_row)
        return cleaned_data
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return []
    except Exception as e:
        print(f"Error processing file: {e}")
        return []

def write_cleaned_csv(data, output_path):
    """Write cleaned data to a new CSV file."""
    if not data:
        return False
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        return True
    except Exception as e:
        print(f"Error writing file: {e}")
        return False
import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    remove_duplicates (bool): Whether to remove duplicate rows
    fill_missing (str or dict): Strategy to fill missing values
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
        elif fill_missing == 'drop':
            cleaned_df = cleaned_df.dropna()
    
    # Remove duplicates
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
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
    
    return True, "Dataset is valid"

# Example usage
if __name__ == "__main__":
    # Create sample data with issues
    sample_data = {
        'id': [1, 2, 2, 3, 4, None],
        'value': [10, None, 20, 30, 40, 50],
        'category': ['A', 'B', 'B', None, 'A', 'C']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nShape:", df.shape)
    
    # Clean the data
    cleaned = clean_dataset(df, remove_duplicates=True, fill_missing={'value': 0})
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nShape:", cleaned.shape)
    
    # Validate the cleaned data
    is_valid, message = validate_dataset(cleaned, required_columns=['id', 'value'], min_rows=3)
    print(f"\nValidation: {is_valid} - {message}")