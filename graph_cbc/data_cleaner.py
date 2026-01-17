
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
            subset: List of column names to consider for identifying duplicates.
                    If None, all columns are used.
            keep: Determines which duplicates to keep.
                  'first' : Keep first occurrence.
                  'last'  : Keep last occurrence.
                  False   : Drop all duplicates.
                  
        Returns:
            Cleaned DataFrame with duplicates removed.
        """
        cleaned_df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed_count = len(self.df) - len(cleaned_df)
        
        if removed_count > 0:
            print(f"Removed {removed_count} duplicate rows.")
            print(f"Original shape: {self.original_shape}")
            print(f"New shape: {cleaned_df.shape}")
        
        return cleaned_df
    
    def fill_missing_values(self, strategy: str = 'mean', fill_value: Optional[float] = None) -> pd.DataFrame:
        """
        Fill missing values in numeric columns.
        
        Args:
            strategy: 'mean', 'median', 'mode', or 'constant'
            fill_value: Value to use when strategy is 'constant'
            
        Returns:
            DataFrame with missing values filled.
        """
        df_filled = self.df.copy()
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df_filled[col].isnull().any():
                if strategy == 'mean':
                    fill_val = df_filled[col].mean()
                elif strategy == 'median':
                    fill_val = df_filled[col].median()
                elif strategy == 'mode':
                    fill_val = df_filled[col].mode()[0]
                elif strategy == 'constant' and fill_value is not None:
                    fill_val = fill_value
                else:
                    continue
                    
                df_filled[col].fillna(fill_val, inplace=True)
                print(f"Filled missing values in column '{col}' using {strategy} strategy.")
        
        return df_filled
    
    def remove_outliers_iqr(self, columns: List[str], multiplier: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers using the Interquartile Range method.
        
        Args:
            columns: List of column names to check for outliers.
            multiplier: IQR multiplier (default 1.5).
            
        Returns:
            DataFrame with outliers removed.
        """
        df_clean = self.df.copy()
        original_len = len(df_clean)
        
        for col in columns:
            if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                df_clean = df_clean[mask]
        
        removed_count = original_len - len(df_clean)
        if removed_count > 0:
            print(f"Removed {removed_count} outliers using IQR method.")
        
        return df_clean

def clean_dataset(file_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Complete data cleaning pipeline for a CSV file.
    
    Args:
        file_path: Path to input CSV file.
        output_path: Path to save cleaned data (optional).
        
    Returns:
        Cleaned DataFrame.
    """
    df = pd.read_csv(file_path)
    cleaner = DataCleaner(df)
    
    print("Starting data cleaning process...")
    print(f"Original data shape: {df.shape}")
    
    df_clean = cleaner.remove_duplicates()
    df_clean = cleaner.fill_missing_values(strategy='median')
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        df_clean = cleaner.remove_outliers_iqr(numeric_cols)
    
    print(f"Final data shape: {df_clean.shape}")
    
    if output_path:
        df_clean.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    return df_clean

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 5, 6],
        'value': [10.5, 20.3, np.nan, 15.7, 1000.0, 1000.0, 12.8],
        'category': ['A', 'B', 'A', 'C', 'A', 'A', 'B']
    })
    
    cleaner = DataCleaner(sample_data)
    cleaned = cleaner.remove_duplicates()
    cleaned = cleaner.fill_missing_values()
    
    print("Sample cleaning completed.")