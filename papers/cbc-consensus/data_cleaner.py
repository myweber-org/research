
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.df.select_dtypes(include=['object']).columns.tolist()
    
    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean' and self.numeric_columns:
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].mean())
        elif strategy == 'median' and self.numeric_columns:
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].median())
        elif strategy == 'mode' and self.categorical_columns:
            for col in self.categorical_columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown')
        elif fill_value is not None:
            self.df = self.df.fillna(fill_value)
        return self
    
    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        return self
    
    def standardize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns:
                mean = self.df[col].mean()
                std = self.df[col].std()
                if std > 0:
                    self.df[col] = (self.df[col] - mean) / std
        return self
    
    def normalize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        return self
    
    def get_cleaned_data(self):
        return self.df.copy()
    
    def get_summary(self):
        summary = {
            'original_shape': self.df.shape,
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': len(self.numeric_columns),
            'categorical_columns': len(self.categorical_columns)
        }
        return summary

def example_usage():
    data = {
        'age': [25, 30, 35, None, 45, 200, 28, 32, None, 29],
        'salary': [50000, 60000, None, 70000, 80000, 90000, 1000000, 55000, None, 65000],
        'department': ['IT', 'HR', 'IT', None, 'Finance', 'IT', 'HR', None, 'Finance', 'IT']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\nSummary:")
    print(df.info())
    
    cleaner = DataCleaner(df)
    cleaner.handle_missing_values(strategy='mean')
    cleaner.remove_outliers_iqr(multiplier=1.5)
    cleaner.standardize_data()
    
    cleaned_df = cleaner.get_cleaned_data()
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaning Summary:")
    print(cleaner.get_summary())

if __name__ == "__main__":
    example_usage()
import pandas as pd
import re

def clean_dataframe(df, text_columns=None, drop_duplicates=True, case_sensitive=False):
    """
    Clean a DataFrame by removing duplicates and standardizing text columns.
    
    Args:
        df: pandas DataFrame to clean
        text_columns: list of column names to standardize (if None, all object columns)
        drop_duplicates: whether to remove duplicate rows
        case_sensitive: whether to preserve case when comparing duplicates
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    # Standardize text columns
    if text_columns is None:
        text_columns = cleaned_df.select_dtypes(include=['object']).columns.tolist()
    
    for col in text_columns:
        if col in cleaned_df.columns:
            # Remove extra whitespace
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
            # Replace multiple spaces with single space
            cleaned_df[col] = cleaned_df[col].apply(lambda x: re.sub(r'\s+', ' ', x))
            # Standardize case if not case sensitive
            if not case_sensitive:
                cleaned_df[col] = cleaned_df[col].str.lower()
    
    # Remove duplicates
    if drop_duplicates:
        if case_sensitive:
            cleaned_df = cleaned_df.drop_duplicates()
        else:
            # For case-insensitive duplicate removal, we need to compare lowercased versions
            temp_df = cleaned_df.copy()
            for col in text_columns:
                if col in temp_df.columns:
                    temp_df[col] = temp_df[col].str.lower()
            mask = ~temp_df.duplicated()
            cleaned_df = cleaned_df[mask].reset_index(drop=True)
    
    return cleaned_df

def validate_email_column(df, email_column):
    """
    Validate email addresses in a DataFrame column.
    
    Args:
        df: pandas DataFrame
        email_column: name of the column containing email addresses
    
    Returns:
        DataFrame with valid emails and validation status
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    result_df = df.copy()
    result_df['email_valid'] = result_df[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notna(x) else False
    )
    
    return result_df

def remove_outliers_iqr(df, numeric_columns=None, threshold=1.5):
    """
    Remove outliers using the Interquartile Range (IQR) method.
    
    Args:
        df: pandas DataFrame
        numeric_columns: list of numeric column names (if None, all numeric columns)
        threshold: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    filtered_df = df.copy()
    
    for col in numeric_columns:
        if col in filtered_df.columns:
            Q1 = filtered_df[col].quantile(0.25)
            Q3 = filtered_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Keep only non-outliers
            filtered_df = filtered_df[
                (filtered_df[col] >= lower_bound) & 
                (filtered_df[col] <= upper_bound)
            ]
    
    return filtered_df.reset_index(drop=True)