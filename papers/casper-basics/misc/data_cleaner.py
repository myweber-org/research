import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = df.select_dtypes(exclude=[np.number]).columns
        
    def detect_outliers_iqr(self, column, threshold=1.5):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        return outliers.index.tolist()
    
    def remove_outliers(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.numeric_columns
            
        outlier_indices = []
        for col in columns:
            if col in self.numeric_columns:
                outlier_indices.extend(self.detect_outliers_iqr(col, threshold))
        
        unique_outliers = list(set(outlier_indices))
        cleaned_df = self.df.drop(index=unique_outliers)
        return cleaned_df, len(unique_outliers)
    
    def impute_missing_median(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
            
        imputed_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns and imputed_df[col].isnull().any():
                median_value = imputed_df[col].median()
                imputed_df[col].fillna(median_value, inplace=True)
        return imputed_df
    
    def impute_missing_mode(self, columns=None):
        if columns is None:
            columns = self.categorical_columns
            
        imputed_df = self.df.copy()
        for col in columns:
            if col in self.categorical_columns and imputed_df[col].isnull().any():
                mode_value = imputed_df[col].mode()[0]
                imputed_df[col].fillna(mode_value, inplace=True)
        return imputed_df
    
    def standardize_numeric(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
            
        standardized_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                mean = standardized_df[col].mean()
                std = standardized_df[col].std()
                if std > 0:
                    standardized_df[col] = (standardized_df[col] - mean) / std
        return standardized_df
    
    def get_summary(self):
        summary = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'numeric_columns': list(self.numeric_columns),
            'categorical_columns': list(self.categorical_columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary
import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_na_method='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): List of column names to check for duplicates.
            If None, uses all columns. Defaults to None.
        fill_na_method (str, optional): Method to fill missing values.
            Options: 'mean', 'median', 'mode', or 'drop'. Defaults to 'mean'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Create a copy to avoid modifying the original DataFrame
    cleaned_df = df.copy()
    
    # Remove duplicates
    if columns_to_check is None:
        columns_to_check = cleaned_df.columns.tolist()
    
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check, keep='first')
    removed_duplicates = initial_rows - len(cleaned_df)
    
    # Handle missing values
    if fill_na_method == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_na_method in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_na_method == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_na_method == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                mode_value = cleaned_df[col].mode()
                if not mode_value.empty:
                    cleaned_df[col] = cleaned_df[col].fillna(mode_value[0])
    
    # Print cleaning summary
    print(f"Removed {removed_duplicates} duplicate rows")
    print(f"Final dataset shape: {cleaned_df.shape}")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of required column names.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'id': [1, 2, 2, 3, 4, 5],
#         'name': ['Alice', 'Bob', 'Bob', 'Charlie', None, 'Eve'],
#         'age': [25, 30, 30, None, 35, 40],
#         'score': [85.5, 90.0, 90.0, 78.5, None, 95.0]
#     }
#     
#     df = pd.DataFrame(data)
#     print("Original DataFrame:")
#     print(df)
#     print("\nCleaning dataset...")
#     
#     cleaned = clean_dataset(df, fill_na_method='mean')
#     print("\nCleaned DataFrame:")
#     print(cleaned)