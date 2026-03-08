
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.numeric_columns
        
        clean_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                Q1 = clean_df[col].quantile(0.25)
                Q3 = clean_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                mask = (clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)
                clean_df = clean_df[mask]
        
        return clean_df
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.numeric_columns
        
        clean_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                z_scores = np.abs(stats.zscore(clean_df[col].dropna()))
                mask = z_scores < threshold
                clean_df = clean_df[mask]
        
        return clean_df
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        normalized_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val != min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
        
        return normalized_df
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        normalized_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                mean_val = normalized_df[col].mean()
                std_val = normalized_df[col].std()
                if std_val > 0:
                    normalized_df[col] = (normalized_df[col] - mean_val) / std_val
        
        return normalized_df
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        filled_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                filled_df[col] = filled_df[col].fillna(filled_df[col].mean())
        
        return filled_df
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        filled_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                filled_df[col] = filled_df[col].fillna(filled_df[col].median())
        
        return filled_df
    
    def get_summary(self):
        summary = {
            'original_shape': self.df.shape,
            'numeric_columns': self.numeric_columns,
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
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
    
    print("Data Summary:")
    summary = cleaner.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\nCleaning with IQR method...")
    cleaned_iqr = cleaner.remove_outliers_iqr()
    print(f"Original shape: {sample_df.shape}, Cleaned shape: {cleaned_iqr.shape}")
    
    print("\nNormalizing data...")
    normalized = cleaner.normalize_minmax()
    print(f"Normalized data sample:\n{normalized.head()}")
import pandas as pd
import numpy as np
from datetime import datetime

def clean_dataframe(df, date_column='date', id_column='id'):
    """
    Clean dataframe by removing duplicates, standardizing dates,
    and filling missing values.
    """
    # Remove exact duplicates
    initial_count = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_count - len(df)
    
    # Standardize date format
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    # Fill missing numeric values with column median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    # Remove rows where ID is null
    if id_column in df.columns:
        df = df.dropna(subset=[id_column])
    
    # Reset index after cleaning
    df = df.reset_index(drop=True)
    
    return df, duplicates_removed

def validate_data(df, required_columns):
    """
    Validate that dataframe contains all required columns
    and has no null values in key fields.
    """
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    validation_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'null_counts': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict()
    }
    
    return validation_report

def export_clean_data(df, output_path, format='csv'):
    """
    Export cleaned data to specified format.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if format == 'csv':
        output_file = f"{output_path}/cleaned_data_{timestamp}.csv"
        df.to_csv(output_file, index=False)
    elif format == 'excel':
        output_file = f"{output_path}/cleaned_data_{timestamp}.xlsx"
        df.to_excel(output_file, index=False)
    elif format == 'parquet':
        output_file = f"{output_path}/cleaned_data_{timestamp}.parquet"
        df.to_parquet(output_file, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return output_file

def main():
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, None],
        'date': ['2023-01-01', '2023-01-02', '2023-01-02', 'invalid', '2023-01-04', '2023-01-05'],
        'value': [10.5, None, 15.3, 20.1, 25.7, 30.2],
        'category': ['A', 'B', 'B', 'C', 'A', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data shape:", df.shape)
    
    # Clean the data
    cleaned_df, removed = clean_dataframe(df)
    print(f"Removed {removed} duplicate rows")
    print("Cleaned data shape:", cleaned_df.shape)
    
    # Validate data
    required_cols = ['id', 'date', 'value']
    try:
        report = validate_data(cleaned_df, required_cols)
        print("Validation report:", report)
    except ValueError as e:
        print(f"Validation error: {e}")
    
    # Export cleaned data
    output_path = './output'
    exported_file = export_clean_data(cleaned_df, output_path, format='csv')
    print(f"Data exported to: {exported_file}")

if __name__ == "__main__":
    main()