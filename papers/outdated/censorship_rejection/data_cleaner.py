import csv
import re

def remove_duplicates(input_file, output_file):
    """Remove duplicate rows from a CSV file."""
    seen = set()
    unique_rows = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            row_tuple = tuple(row)
            if row_tuple not in seen:
                seen.add(row_tuple)
                unique_rows.append(row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(unique_rows)
    
    return len(unique_rows)

def clean_numeric_columns(input_file, output_file, columns):
    """Clean numeric columns by removing non-numeric characters."""
    cleaned_data = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        cleaned_data.append(fieldnames)
        
        for row in reader:
            cleaned_row = []
            for field in fieldnames:
                value = row[field]
                if field in columns:
                    # Remove all non-numeric characters except decimal point
                    value = re.sub(r'[^\d.]', '', value)
                    # Handle empty results
                    if value == '':
                        value = '0'
                cleaned_row.append(value)
            cleaned_data.append(cleaned_row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(cleaned_data)
    
    return len(cleaned_data) - 1

def validate_email_format(email):
    """Validate email format using regex."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def filter_valid_emails(input_file, output_file, email_column):
    """Filter rows with valid email addresses."""
    valid_rows = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        valid_rows.append(fieldnames)
        
        for row in reader:
            if validate_email_format(row[email_column]):
                valid_rows.append([row[field] for field in fieldnames])
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(valid_rows)
    
    return len(valid_rows) - 1

def merge_csv_files(file_list, output_file):
    """Merge multiple CSV files into one."""
    all_data = []
    headers_set = set()
    
    for file_path in file_list:
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
            headers_set.add(tuple(headers))
            
            if not all_data:
                all_data.append(headers)
            
            for row in reader:
                all_data.append(row)
    
    if len(headers_set) > 1:
        raise ValueError("CSV files have different headers")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(all_data)
    
    return len(all_data) - 1
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_clean = self.df.copy()
        for col in columns:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        self.df = df_clean
        return self
        
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_clean = self.df.copy()
        for col in columns:
            if col in df_clean.columns:
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                df_clean = df_clean[(z_scores < threshold) | df_clean[col].isna()]
        
        self.df = df_clean
        return self
        
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_norm = self.df.copy()
        for col in columns:
            if col in df_norm.columns:
                min_val = df_norm[col].min()
                max_val = df_norm[col].max()
                if max_val != min_val:
                    df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        
        self.df = df_norm
        return self
        
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_norm = self.df.copy()
        for col in columns:
            if col in df_norm.columns:
                mean_val = df_norm[col].mean()
                std_val = df_norm[col].std()
                if std_val > 0:
                    df_norm[col] = (df_norm[col] - mean_val) / std_val
        
        self.df = df_norm
        return self
        
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_filled = self.df.copy()
        for col in columns:
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
        
        self.df = df_filled
        return self
        
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_filled = self.df.copy()
        for col in columns:
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].fillna(df_filled[col].median())
        
        self.df = df_filled
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]
        
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.get_removed_count(),
            'missing_values': self.df.isnull().sum().sum()
        }
        return summary
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    
    if drop_duplicates:
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")
    
    if df.isnull().sum().any():
        print("Handling missing values...")
        if fill_missing == 'mean':
            df = df.fillna(df.mean(numeric_only=True))
        elif fill_missing == 'median':
            df = df.fillna(df.median(numeric_only=True))
        elif fill_missing == 'mode':
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        elif fill_missing == 'drop':
            df = df.dropna()
        else:
            raise ValueError("Invalid fill_missing option")
    
    print(f"Data cleaned. Original shape: {original_shape}, New shape: {df.shape}")
    return df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the DataFrame structure and content.
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if len(df) < min_rows:
        raise ValueError(f"DataFrame has fewer than {min_rows} rows")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

def main():
    # Example usage
    data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, np.nan, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'C', None, 'A']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataframe(df, drop_duplicates=True, fill_missing='mode')
    
    try:
        validate_data(cleaned_df, required_columns=['id', 'value'], min_rows=3)
        print("Data validation passed")
    except ValueError as e:
        print(f"Data validation failed: {e}")
    
    return cleaned_df

if __name__ == "__main__":
    result = main()
    print("\nCleaned DataFrame:")
    print(result)
import pandas as pd

def clean_dataframe(df, drop_na=True, rename_columns=True):
    """
    Clean a pandas DataFrame by removing null values and standardizing column names.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_na (bool): Whether to drop rows with null values. Default is True.
        rename_columns (bool): Whether to rename columns to lowercase with underscores. Default is True.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_na:
        cleaned_df = cleaned_df.dropna()
    
    if rename_columns:
        cleaned_df.columns = (
            cleaned_df.columns
            .str.lower()
            .str.replace(' ', '_')
            .str.replace('[^a-z0-9_]', '', regex=True)
        )
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for required columns and data types.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names. Default is None.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    if df.empty:
        print("DataFrame is empty")
        return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'First Name': ['Alice', 'Bob', None, 'Charlie'],
        'Last Name': ['Smith', 'Johnson', 'Williams', None],
        'Age': [25, 30, 35, 40]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataframe(df)
    print(cleaned)
    
    is_valid = validate_dataframe(cleaned, required_columns=['first_name', 'last_name', 'age'])
    print(f"\nDataFrame validation: {is_valid}")
import pandas as pd
import numpy as np

def clean_dataset(df, column_mapping=None, drop_duplicates=True):
    """
    Clean a pandas DataFrame by standardizing column names and removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    column_mapping (dict): Optional dictionary mapping old column names to new ones
    drop_duplicates (bool): Whether to remove duplicate rows
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Standardize column names
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    # Convert column names to lowercase and replace spaces with underscores
    cleaned_df.columns = cleaned_df.columns.str.lower().str.replace(' ', '_')
    
    # Remove duplicate rows if requested
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    # Convert string columns to proper case for consistency
    for col in cleaned_df.select_dtypes(include=['object']).columns:
        cleaned_df[col] = cleaned_df[col].apply(
            lambda x: x.title() if isinstance(x, str) else x
        )
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of column names that must be present
    
    Returns:
    dict: Dictionary containing validation results
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'null_counts': {},
        'data_types': {}
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['is_valid'] = False
            validation_results['missing_columns'] = missing
    
    # Count null values in each column
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            validation_results['null_counts'][col] = null_count
        
        validation_results['data_types'][col] = str(df[col].dtype)
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Customer Name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson'],
        'Order Date': ['2023-01-15', '2023-01-16', '2023-01-15', '2023-01-17'],
        'Product': ['laptop', 'MOUSE', 'laptop', 'keyboard'],
        'Price': [999.99, 49.99, 999.99, 79.99]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df)
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    validation = validate_data(cleaned, required_columns=['customer_name', 'price'])
    print("Validation Results:")
    print(validation)
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None):
    """
    Clean a pandas DataFrame by removing duplicate rows and normalizing string columns.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    if columns_to_clean is None:
        # Automatically detect string columns
        columns_to_clean = df_cleaned.select_dtypes(include=['object']).columns.tolist()
    
    for col in columns_to_clean:
        if col in df_cleaned.columns and df_cleaned[col].dtype == 'object':
            # Normalize strings: strip whitespace, convert to lowercase, remove extra spaces
            df_cleaned[col] = df_cleaned[col].astype(str).apply(
                lambda x: re.sub(r'\s+', ' ', x.strip().lower())
            )
    
    return df_cleaned

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    Returns a DataFrame with a new boolean column indicating valid emails.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['email_valid'] = df[email_column].astype(str).str.match(email_pattern)
    
    return df

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a numeric column using the Interquartile Range method.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)