import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_data = data[(z_scores < threshold) | (data[column].isna())]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_data(data, column):
    """
    Standardize data using Z-score normalization
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_missing_values(data, strategy='mean'):
    """
    Handle missing values with different strategies
    """
    cleaned_data = data.copy()
    
    for column in cleaned_data.columns:
        if cleaned_data[column].isnull().any():
            if strategy == 'mean':
                fill_value = cleaned_data[column].mean()
            elif strategy == 'median':
                fill_value = cleaned_data[column].median()
            elif strategy == 'mode':
                fill_value = cleaned_data[column].mode()[0]
            elif strategy == 'ffill':
                cleaned_data[column] = cleaned_data[column].ffill()
                continue
            elif strategy == 'bfill':
                cleaned_data[column] = cleaned_data[column].bfill()
                continue
            else:
                fill_value = 0
            
            cleaned_data[column] = cleaned_data[column].fillna(fill_value)
    
    return cleaned_data

def validate_data_types(data, expected_types):
    """
    Validate that columns have expected data types
    """
    validation_results = {}
    
    for column, expected_type in expected_types.items():
        if column in data.columns:
            actual_type = str(data[column].dtype)
            validation_results[column] = {
                'expected': expected_type,
                'actual': actual_type,
                'valid': actual_type == expected_type
            }
    
    return validation_results

def create_data_summary(data):
    """
    Create a comprehensive summary of the dataset
    """
    summary = {
        'shape': data.shape,
        'columns': list(data.columns),
        'dtypes': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'unique_values': {col: data[col].nunique() for col in data.columns},
        'basic_stats': data.describe().to_dict()
    }
    
    return summary
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None):
    """
    Clean CSV data by handling missing values and standardizing columns.
    """
    try:
        df = pd.read_csv(input_path)
        
        # Remove completely empty rows and columns
        df.dropna(how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        
        # Fill numeric missing values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Remove duplicate rows
        df.drop_duplicates(inplace=True)
        
        # Reset index after cleaning
        df.reset_index(drop=True, inplace=True)
        
        # Save cleaned data
        if output_path is None:
            input_file = Path(input_path)
            output_path = input_file.parent / f"cleaned_{input_file.name}"
        
        df.to_csv(output_path, index=False)
        
        print(f"Data cleaning completed. Cleaned data saved to: {output_path}")
        print(f"Original shape: {df.shape}")
        print(f"Missing values after cleaning:\n{df.isnull().sum()}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return None
    except pd.errors.EmptyDataError:
        print("Error: Input file is empty")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df):
    """
    Validate dataframe for common data quality issues.
    """
    if df is None or df.empty:
        print("DataFrame is empty or None")
        return False
    
    checks = {
        'has_null_values': df.isnull().sum().sum() == 0,
        'has_duplicates': not df.duplicated().any(),
        'has_infinite_values': np.isfinite(df.select_dtypes(include=[np.number])).all().all(),
        'has_empty_strings': (df.select_dtypes(include=['object']) == '').sum().sum() == 0
    }
    
    print("Data Validation Results:")
    for check_name, check_result in checks.items():
        status = "PASS" if check_result else "FAIL"
        print(f"  {check_name}: {status}")
    
    return all(checks.values())

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Name': ['Alice', 'Bob', 'Charlie', None, 'Eve'],
        'Age': [25, None, 30, 35, 40],
        'Salary': [50000, 60000, None, 70000, 80000],
        'Department': ['IT', 'HR', 'IT', None, 'Finance']
    }
    
    # Create test dataframe
    test_df = pd.DataFrame(sample_data)
    test_csv = "test_data.csv"
    test_df.to_csv(test_csv, index=False)
    
    # Clean the data
    cleaned_df = clean_csv_data(test_csv)
    
    # Validate cleaned data
    if cleaned_df is not None:
        validate_dataframe(cleaned_df)
    
    # Clean up test file
    import os
    if os.path.exists(test_csv):
        os.remove(test_csv)
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

if __name__ == "__main__":
    sample_data = [3, 1, 2, 1, 4, 3, 5, 2]
    cleaned = remove_duplicates_preserve_order(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")