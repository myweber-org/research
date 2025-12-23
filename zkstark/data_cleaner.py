
import pandas as pd
import numpy as np
from typing import List, Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape

    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = 'first') -> pd.DataFrame:
        """
        Remove duplicate rows from the DataFrame.
        
        Args:
            subset: Column labels to consider for identifying duplicates.
            keep: Determines which duplicates to keep.
        
        Returns:
            Cleaned DataFrame with duplicates removed.
        """
        cleaned_df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed_count = len(self.df) - len(cleaned_df)
        
        if removed_count > 0:
            print(f"Removed {removed_count} duplicate rows.")
            print(f"Original shape: {self.original_shape}")
            print(f"New shape: {cleaned_df.shape}")
        
        self.df = cleaned_df
        return self.df

    def fill_missing_values(self, strategy: str = 'mean', fill_value: Optional[float] = None) -> pd.DataFrame:
        """
        Fill missing values in numeric columns.
        
        Args:
            strategy: Method to use for filling ('mean', 'median', 'mode', or 'constant').
            fill_value: Value to use when strategy is 'constant'.
        
        Returns:
            DataFrame with missing values filled.
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if strategy == 'mean':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == 'median':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif strategy == 'mode':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mode().iloc[0])
        elif strategy == 'constant' and fill_value is not None:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(fill_value)
        else:
            raise ValueError("Invalid strategy or missing fill_value for constant strategy")
        
        print(f"Filled missing values using {strategy} strategy.")
        return self.df

    def remove_outliers_iqr(self, columns: List[str], multiplier: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers using the Interquartile Range method.
        
        Args:
            columns: List of column names to check for outliers.
            multiplier: IQR multiplier for outlier detection.
        
        Returns:
            DataFrame with outliers removed.
        """
        original_len = len(self.df)
        
        for col in columns:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        
        removed_count = original_len - len(self.df)
        if removed_count > 0:
            print(f"Removed {removed_count} outliers using IQR method.")
        
        return self.df

    def get_cleaned_data(self) -> pd.DataFrame:
        """
        Return the cleaned DataFrame.
        
        Returns:
            Final cleaned DataFrame.
        """
        return self.df.copy()

def create_sample_data() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    data = {
        'id': [1, 2, 3, 4, 5, 1, 2, 6],
        'value': [10.5, 20.3, 15.7, np.nan, 12.8, 10.5, 20.3, 100.0],
        'category': ['A', 'B', 'A', 'B', 'A', 'A', 'B', 'C']
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Original DataFrame:")
    print(sample_df)
    print("\n" + "="*50 + "\n")
    
    cleaner = DataCleaner(sample_df)
    cleaner.remove_duplicates(subset=['id'])
    cleaner.fill_missing_values(strategy='mean')
    cleaner.remove_outliers_iqr(columns=['value'])
    
    cleaned_df = cleaner.get_cleaned_data()
    print("\nCleaned DataFrame:")
    print(cleaned_df)
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_data(input_file, output_file):
    df = load_dataset(input_file)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
    
    for col in numeric_columns:
        df = normalize_column(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    
    return df

if __name__ == "__main__":
    cleaned_df = clean_data('raw_data.csv', 'cleaned_data.csv')import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    Returns a cleaned DataFrame.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    cleaned_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return cleaned_data

def normalize_column(data, column, method='minmax'):
    """
    Normalize a column in the DataFrame.
    Supports 'minmax' and 'zscore' normalization methods.
    Returns a new DataFrame with the normalized column.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    data_copy = data.copy()
    
    if method == 'minmax':
        min_val = data_copy[column].min()
        max_val = data_copy[column].max()
        if max_val == min_val:
            data_copy[column] = 0.5
        else:
            data_copy[column] = (data_copy[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = data_copy[column].mean()
        std_val = data_copy[column].std()
        if std_val == 0:
            data_copy[column] = 0
        else:
            data_copy[column] = (data_copy[column] - mean_val) / std_val
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return data_copy

def clean_dataset(data, numeric_columns, outlier_threshold=1.5, normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    Removes outliers and normalizes specified numeric columns.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_data = data.copy()
    
    for col in numeric_columns:
        if col in cleaned_data.columns:
            # Remove outliers
            Q1 = cleaned_data[col].quantile(0.25)
            Q3 = cleaned_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - outlier_threshold * IQR
            upper_bound = Q3 + outlier_threshold * IQR
            
            mask = (cleaned_data[col] >= lower_bound) & (cleaned_data[col] <= upper_bound)
            cleaned_data = cleaned_data[mask]
            
            # Normalize
            cleaned_data = normalize_column(cleaned_data, col, method=normalize_method)
    
    return cleaned_data.reset_index(drop=True)

def validate_dataframe(data, required_columns=None, allow_nan=False):
    """
    Validate DataFrame structure and content.
    """
    if not isinstance(data, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if not allow_nan and data.isnull().any().any():
        return False, "DataFrame contains NaN values"
    
    return True, "DataFrame validation passed"