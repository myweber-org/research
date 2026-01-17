import pandas as pd
import numpy as np
from typing import Optional, Dict, List

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def handle_missing_values(self, strategy: str = 'mean', columns: Optional[List[str]] = None) -> 'DataCleaner':
        if columns is None:
            columns = self.df.columns
            
        for col in columns:
            if col in self.df.columns:
                if strategy == 'mean' and pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif strategy == 'mode':
                    self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else np.nan, inplace=True)
                elif strategy == 'drop':
                    self.df.dropna(subset=[col], inplace=True)
                else:
                    self.df[col].fillna(method='ffill', inplace=True)
        return self
    
    def convert_dtypes(self, type_mapping: Dict[str, str]) -> 'DataCleaner':
        for col, dtype in type_mapping.items():
            if col in self.df.columns:
                try:
                    if dtype == 'datetime':
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    elif dtype == 'numeric':
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    elif dtype == 'category':
                        self.df[col] = self.df[col].astype('category')
                    else:
                        self.df[col] = self.df[col].astype(dtype)
                except (ValueError, TypeError):
                    continue
        return self
    
    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = 'first') -> 'DataCleaner':
        self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        return self
    
    def normalize_column(self, column: str, method: str = 'minmax') -> 'DataCleaner':
        if column in self.df.columns and pd.api.types.is_numeric_dtype(self.df[column]):
            if method == 'minmax':
                min_val = self.df[column].min()
                max_val = self.df[column].max()
                if max_val > min_val:
                    self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = self.df[column].mean()
                std_val = self.df[column].std()
                if std_val > 0:
                    self.df[column] = (self.df[column] - mean_val) / std_val
        return self
    
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
    
    def get_cleaning_report(self) -> Dict:
        return {
            'original_shape': self.original_shape,
            'cleaned_shape': self.df.shape,
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.astype(str).to_dict()
        }

def load_and_clean_csv(filepath: str, cleaning_steps: Optional[Dict] = None) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    cleaner = DataCleaner(df)
    
    if cleaning_steps:
        if 'missing_strategy' in cleaning_steps:
            cleaner.handle_missing_values(
                strategy=cleaning_steps['missing_strategy'],
                columns=cleaning_steps.get('missing_columns')
            )
        if 'type_mapping' in cleaning_steps:
            cleaner.convert_dtypes(cleaning_steps['type_mapping'])
        if 'remove_dups' in cleaning_steps and cleaning_steps['remove_dups']:
            cleaner.remove_duplicates(subset=cleaning_steps.get('dup_subset'))
        if 'normalize' in cleaning_steps:
            for norm_config in cleaning_steps['normalize']:
                cleaner.normalize_column(**norm_config)
    
    return cleaner.get_cleaned_data()import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        drop_duplicates: Whether to drop duplicate rows
        fill_missing: Whether to fill missing values
        fill_value: Value to use for filling missing data
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate that the DataFrame meets basic requirements.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of columns that must be present
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Data validation passed"

def remove_outliers(df, column, threshold=3):
    """
    Remove outliers from a specific column using z-score method.
    
    Args:
        df: pandas DataFrame
        column: Column name to check for outliers
        threshold: Z-score threshold for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    from scipy import stats
    import numpy as np
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    mask = z_scores < threshold
    
    return df[mask].reset_index(drop=True)import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    if max_val == min_val:
        return df[column].apply(lambda x: 0.5)
    return df[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df):
    required_checks = [
        (lambda x: isinstance(x, pd.DataFrame), "Input must be a pandas DataFrame"),
        (lambda x: not x.empty, "DataFrame cannot be empty"),
        (lambda x: x.isnull().sum().sum() == 0, "DataFrame contains null values")
    ]
    for check_func, error_msg in required_checks:
        if not check_func(df):
            raise ValueError(error_msg)
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 50),
        'feature_b': np.random.exponential(2, 50),
        'category': np.random.choice(['X', 'Y', 'Z'], 50)
    })
    print("Original shape:", sample_data.shape)
    cleaned = clean_dataset(sample_data, ['feature_a', 'feature_b'])
    print("Cleaned shape:", cleaned.shape)
    print("Sample normalized values:")
    print(cleaned[['feature_a', 'feature_b']].head())import pandas as pd
import numpy as np
from scipy import stats

def load_data(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    """Remove outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    """Normalize column using min-max scaling."""
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(input_file, output_file):
    """Main cleaning pipeline."""
    df = load_data(input_file)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        df = remove_outliers_iqr(df, col)
    
    for col in numeric_cols:
        df = normalize_column(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    
    return df

if __name__ == "__main__":
    cleaned_df = clean_dataset('raw_data.csv', 'cleaned_data.csv')