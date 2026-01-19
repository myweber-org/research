
def remove_duplicates_preserve_order(seq):
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    
    if drop_duplicates:
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")
    
    if fill_missing:
        for column in df.columns:
            if df[column].isnull().any():
                if fill_missing == 'mean' and pd.api.types.is_numeric_dtype(df[column]):
                    df[column] = df[column].fillna(df[column].mean())
                elif fill_missing == 'median' and pd.api.types.is_numeric_dtype(df[column]):
                    df[column] = df[column].fillna(df[column].median())
                elif fill_missing == 'mode':
                    df[column] = df[column].fillna(df[column].mode()[0])
                elif fill_missing == 'ffill':
                    df[column] = df[column].fillna(method='ffill')
                elif fill_missing == 'bfill':
                    df[column] = df[column].fillna(method='bfill')
                else:
                    df[column] = df[column].fillna(0)
    
    print(f"Dataset cleaned: {original_shape} -> {df.shape}")
    return df

def validate_dataset(df, required_columns=None):
    """
    Validate dataset structure and content.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    validation_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    return validation_report

def main():
    # Example usage
    data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10, 20, 20, np.nan, 40, 50],
        'category': ['A', 'B', 'B', 'C', None, 'A']
    }
    
    df = pd.DataFrame(data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50)
    
    # Clean the dataset
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    # Validate the dataset
    report = validate_dataset(cleaned_df, required_columns=['id', 'value'])
    print("\nValidation report:")
    for key, value in report.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
    def detect_outliers_iqr(self, column, threshold=1.5):
        if column not in self.numeric_columns:
            return []
        
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
        self.df = self.df.drop(index=unique_outliers)
        return len(unique_outliers)
    
    def impute_missing_mean(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns and self.df[col].isnull().any():
                mean_val = self.df[col].mean()
                self.df[col].fillna(mean_val, inplace=True)
    
    def impute_missing_median(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns and self.df[col].isnull().any():
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
    
    def drop_high_missing(self, threshold=0.3):
        missing_ratios = self.df.isnull().sum() / len(self.df)
        columns_to_drop = missing_ratios[missing_ratios > threshold].index.tolist()
        self.df = self.df.drop(columns=columns_to_drop)
        return columns_to_drop
    
    def get_cleaned_data(self):
        return self.df.copy()
    
    def summary(self):
        summary_data = {
            'original_rows': len(self.df),
            'original_columns': len(self.df.columns),
            'numeric_columns': len(self.numeric_columns),
            'missing_values': self.df.isnull().sum().sum(),
            'missing_percentage': (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        }
        return summary_data

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 200, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[np.random.choice(1000, 50), 'feature_a'] = np.nan
    df.loc[np.random.choice(1000, 30), 'feature_b'] = np.nan
    
    outlier_indices = np.random.choice(1000, 20)
    df.loc[outlier_indices, 'feature_c'] = df['feature_c'].max() * 5
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Initial summary:")
    print(cleaner.summary())
    
    removed = cleaner.remove_outliers(['feature_c'])
    print(f"Removed {removed} outliers")
    
    cleaner.impute_missing_mean(['feature_a'])
    cleaner.impute_missing_median(['feature_b'])
    
    print("\nFinal summary:")
    print(cleaner.summary())
    
    cleaned_df = cleaner.get_cleaned_data()
    print(f"\nCleaned data shape: {cleaned_df.shape}")