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
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        self.df = df_clean
        removed = self.original_shape[0] - self.df.shape[0]
        print(f"Removed {removed} outliers using IQR method")
        return self
        
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        print("Applied Min-Max normalization")
        return self
        
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
        
        print("Applied Z-score standardization")
        return self
        
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
        
        print("Filled missing values with median")
        return self
        
    def get_cleaned_data(self):
        return self.df.copy()
        
    def summary(self):
        print(f"Original shape: {self.original_shape}")
        print(f"Cleaned shape: {self.df.shape}")
        print(f"Removed rows: {self.original_shape[0] - self.df.shape[0]}")
        print(f"Removed columns: {self.original_shape[1] - self.df.shape[1]}")
        print("\nData types:")
        print(self.df.dtypes.value_counts())
        print("\nMissing values:")
        print(self.df.isnull().sum().sum())

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 200, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 50), 'feature_a'] = np.nan
    df.loc[np.random.choice(df.index, 30), 'feature_b'] = 9999
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    cleaned_df = (cleaner
                 .fill_missing_median(['feature_a', 'feature_b'])
                 .remove_outliers_iqr(['feature_a', 'feature_b', 'feature_c'])
                 .standardize_zscore(['feature_a', 'feature_b'])
                 .normalize_minmax(['feature_c'])
                 .get_cleaned_data())
    
    cleaner.summary()
    print(f"\nCleaned data preview:\n{cleaned_df.head()}")import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file, missing_strategy='mean'):
    """
    Clean a CSV file by handling missing values and removing duplicates.
    """
    try:
        df = pd.read_csv(input_file)
        
        # Remove duplicate rows
        df_cleaned = df.drop_duplicates()
        
        # Handle missing values
        if missing_strategy == 'mean':
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(
                df_cleaned[numeric_cols].mean()
            )
        elif missing_strategy == 'median':
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(
                df_cleaned[numeric_cols].median()
            )
        elif missing_strategy == 'drop':
            df_cleaned = df_cleaned.dropna()
        else:
            raise ValueError("Invalid missing_strategy. Use 'mean', 'median', or 'drop'.")
        
        # Save cleaned data
        df_cleaned.to_csv(output_file, index=False)
        print(f"Data cleaned successfully. Saved to {output_file}")
        print(f"Original rows: {len(df)}, Cleaned rows: {len(df_cleaned)}")
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Example usage
    cleaned_data = clean_csv_data('raw_data.csv', 'cleaned_data.csv', 'mean')