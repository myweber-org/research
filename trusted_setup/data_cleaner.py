
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_clean = self.df.copy()
        for col in columns:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                df_clean = df_clean[mask]
        
        removed_count = self.original_shape[0] - df_clean.shape[0]
        self.df = df_clean
        return removed_count
    
    def normalize_data(self, columns=None, method='zscore'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in df_normalized.columns:
                if method == 'zscore':
                    df_normalized[col] = stats.zscore(df_normalized[col])
                elif method == 'minmax':
                    min_val = df_normalized[col].min()
                    max_val = df_normalized[col].max()
                    if max_val != min_val:
                        df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
                elif method == 'robust':
                    median = df_normalized[col].median()
                    iqr = df_normalized[col].quantile(0.75) - df_normalized[col].quantile(0.25)
                    if iqr != 0:
                        df_normalized[col] = (df_normalized[col] - median) / iqr
        
        self.df = df_normalized
        return self.df
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_filled = self.df.copy()
        for col in columns:
            if col in df_filled.columns and df_filled[col].isnull().any():
                if strategy == 'mean':
                    fill_value = df_filled[col].mean()
                elif strategy == 'median':
                    fill_value = df_filled[col].median()
                elif strategy == 'mode':
                    fill_value = df_filled[col].mode()[0]
                elif strategy == 'drop':
                    df_filled = df_filled.dropna(subset=[col])
                    continue
                else:
                    fill_value = 0
                
                df_filled[col] = df_filled[col].fillna(fill_value)
        
        self.df = df_filled
        return self.df
    
    def get_cleaned_data(self):
        return self.df.copy()
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'missing_values': self.df.isnull().sum().sum()
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.uniform(0, 1, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[10:15, 'feature_a'] = np.nan
    df.loc[5, 'feature_b'] = 1000
    df.loc[95, 'feature_b'] = -500
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Initial data shape:", cleaner.df.shape)
    print("Missing values:", cleaner.df.isnull().sum().sum())
    
    removed = cleaner.remove_outliers_iqr(['feature_a', 'feature_b'])
    print(f"Removed {removed} outliers")
    
    cleaner.handle_missing_values(strategy='median')
    print("Missing values after handling:", cleaner.df.isnull().sum().sum())
    
    cleaner.normalize_data(method='zscore')
    
    cleaned_df = cleaner.get_cleaned_data()
    summary = cleaner.get_summary()
    
    print("\nCleaning Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\nFirst 5 rows of cleaned data:")
    print(cleaned_df.head())
import pandas as pd
import numpy as np

def clean_dataset(df, drop_threshold=0.5, fill_strategy='median'):
    """
    Clean a pandas DataFrame by handling missing values and standardizing columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_threshold (float): Threshold for dropping columns with too many nulls
    fill_strategy (str): Strategy for filling missing values ('median', 'mean', 'mode')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Standardize column names
    cleaned_df.columns = cleaned_df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Identify columns with excessive missing values
    null_percentage = cleaned_df.isnull().mean()
    columns_to_drop = null_percentage[null_percentage > drop_threshold].index.tolist()
    
    if columns_to_drop:
        print(f"Dropping columns with >{drop_threshold*100}% nulls: {columns_to_drop}")
        cleaned_df = cleaned_df.drop(columns=columns_to_drop)
    
    # Handle remaining missing values
    for column in cleaned_df.columns:
        if cleaned_df[column].isnull().any():
            if cleaned_df[column].dtype in ['int64', 'float64']:
                if fill_strategy == 'median':
                    fill_value = cleaned_df[column].median()
                elif fill_strategy == 'mean':
                    fill_value = cleaned_df[column].mean()
                else:
                    fill_value = cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else 0
                cleaned_df[column].fillna(fill_value, inplace=True)
            else:
                # For categorical/text columns, fill with mode or 'unknown'
                mode_values = cleaned_df[column].mode()
                fill_value = mode_values[0] if not mode_values.empty else 'unknown'
                cleaned_df[column].fillna(fill_value, inplace=True)
    
    # Remove duplicate rows
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    duplicates_removed = initial_rows - len(cleaned_df)
    
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate rows")
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of columns that must be present
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if not isinstance(df, pd.DataFrame):
        validation_results['is_valid'] = False
        validation_results['errors'].append("Input is not a pandas DataFrame")
        return validation_results
    
    if df.empty:
        validation_results['warnings'].append("DataFrame is empty")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
    
    # Check for completely empty columns
    empty_columns = df.columns[df.isnull().all()].tolist()
    if empty_columns:
        validation_results['warnings'].append(f"Completely empty columns: {empty_columns}")
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Customer ID': [1, 2, 3, 4, 5],
        'Order Value': [100.0, 200.0, np.nan, 400.0, 500.0],
        'Product Category': ['A', 'B', 'A', np.nan, 'C'],
        'Empty Column': [np.nan, np.nan, np.nan, np.nan, np.nan]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, drop_threshold=0.3, fill_strategy='median')
    print("Cleaned DataFrame:")
    print(cleaned)
    
    validation = validate_dataframe(cleaned, required_columns=['customer_id', 'order_value'])
    print("\nValidation Results:")
    print(validation)