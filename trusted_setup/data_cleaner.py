import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    """
    Remove outliers using IQR method.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Main function to clean dataset by removing outliers and normalizing numeric columns.
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize_method == 'minmax':
            cleaned_df = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 200),
        'feature2': np.random.exponential(50, 200),
        'category': np.random.choice(['A', 'B', 'C'], 200)
    })
    
    cleaned = clean_dataset(
        sample_data, 
        numeric_columns=['feature1', 'feature2'],
        outlier_method='iqr',
        normalize_method='zscore'
    )
    
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    print(cleaned.head())
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataset(file_path):
    data = pd.read_csv(file_path)
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        original_len = len(data)
        data = remove_outliers_iqr(data, col)
        removed_count = original_len - len(data)
        print(f"Removed {removed_count} outliers from column '{col}'")
    
    cleaned_file_path = file_path.replace('.csv', '_cleaned.csv')
    data.to_csv(cleaned_file_path, index=False)
    print(f"Cleaned data saved to: {cleaned_file_path}")
    return data

if __name__ == "__main__":
    input_file = "raw_data.csv"
    try:
        cleaned_data = clean_dataset(input_file)
        print(f"Original data shape: {pd.read_csv(input_file).shape}")
        print(f"Cleaned data shape: {cleaned_data.shape}")
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
    except Exception as e:
        print(f"An error occurred: {str(e)}")import pandas as pd
import numpy as np
from scipy import stats

def clean_dataset(df, z_threshold=3, strategy='mean'):
    """
    Clean dataset by handling missing values and removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    z_threshold (float): Z-score threshold for outlier detection
    strategy (str): Strategy for missing value imputation ('mean', 'median', 'mode')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Handle missing values
    if strategy == 'mean':
        df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
    elif strategy == 'median':
        df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
    elif strategy == 'mode':
        df_clean = df_clean.fillna(df_clean.mode().iloc[0])
    
    # Remove outliers using Z-score method
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(df_clean[numeric_cols]))
    
    # Create mask for outliers
    outlier_mask = (z_scores < z_threshold).all(axis=1)
    df_clean = df_clean[outlier_mask].reset_index(drop=True)
    
    return df_clean

def validate_data(df, required_columns=None, min_rows=10):
    """
    Validate dataset structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, message)
    """
    if len(df) < min_rows:
        return False, f"Dataset has less than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        inf_mask = np.isinf(df[numeric_cols]).any().any()
        if inf_mask:
            return False, "Dataset contains infinite values"
    
    return True, "Dataset validation passed"

def normalize_data(df, method='minmax'):
    """
    Normalize numerical columns in the dataset.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    method (str): Normalization method ('minmax' or 'standard')
    
    Returns:
    pd.DataFrame: Normalized DataFrame
    """
    df_norm = df.copy()
    numeric_cols = df_norm.select_dtypes(include=[np.number]).columns
    
    if method == 'minmax':
        for col in numeric_cols:
            col_min = df_norm[col].min()
            col_max = df_norm[col].max()
            if col_max != col_min:
                df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min)
    
    elif method == 'standard':
        for col in numeric_cols:
            col_mean = df_norm[col].mean()
            col_std = df_norm[col].std()
            if col_std != 0:
                df_norm[col] = (df_norm[col] - col_mean) / col_std
    
    return df_norm

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'feature1': [1, 2, 3, 4, 5, 100, 7, 8, 9, 10],
        'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'feature3': [1.1, 2.2, np.nan, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0]
    }
    
    df_sample = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df_sample)
    print("\n" + "="*50)
    
    # Clean the data
    df_cleaned = clean_dataset(df_sample, z_threshold=2.5, strategy='median')
    print("Cleaned dataset:")
    print(df_cleaned)
    print("\n" + "="*50)
    
    # Validate the cleaned data
    is_valid, message = validate_data(df_cleaned, min_rows=5)
    print(f"Validation result: {is_valid}")
    print(f"Validation message: {message}")
    print("\n" + "="*50)
    
    # Normalize the cleaned data
    df_normalized = normalize_data(df_cleaned, method='minmax')
    print("Normalized dataset:")
    print(df_normalized)
import pandas as pd
import numpy as np
import re

def clean_csv_data(input_file, output_file):
    """
    Clean and preprocess CSV data by handling missing values,
    standardizing formats, and removing duplicates.
    """
    try:
        df = pd.read_csv(input_file)
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna('unknown')
        
        # Clean text data
        def clean_text(text):
            if isinstance(text, str):
                text = re.sub(r'\s+', ' ', text)
                text = text.strip()
                return text
            return text
        
        for col in categorical_cols:
            df[col] = df[col].apply(clean_text)
        
        # Remove outliers using IQR method for numeric columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Data cleaning completed. Cleaned data saved to {output_file}")
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except pd.errors.EmptyDataError:
        print("Error: Input file is empty.")
        return False
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return False

def validate_data(file_path):
    """
    Validate the cleaned data file.
    """
    try:
        df = pd.read_csv(file_path)
        
        # Check for remaining missing values
        missing_values = df.isnull().sum().sum()
        
        # Check data types
        data_types = df.dtypes
        
        # Basic statistics
        stats = df.describe()
        
        print(f"Data validation completed:")
        print(f"Total rows: {len(df)}")
        print(f"Total columns: {len(df.columns)}")
        print(f"Missing values: {missing_values}")
        print(f"Data types:\n{data_types}")
        
        return {
            'row_count': len(df),
            'column_count': len(df.columns),
            'missing_values': missing_values,
            'data_types': data_types.to_dict()
        }
        
    except Exception as e:
        print(f"Error during data validation: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    # Clean the data
    success = clean_csv_data(input_csv, output_csv)
    
    # Validate if cleaning was successful
    if success:
        validation_results = validate_data(output_csv)
        if validation_results:
            print("Data cleaning and validation completed successfully.")