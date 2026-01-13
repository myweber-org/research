
import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None, subset=None):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str, optional): Path to output CSV file. 
                                     If None, overwrites input file.
        subset (list, optional): Columns to consider for duplicates.
    """
    try:
        df = pd.read_csv(input_file)
        
        if subset:
            df_clean = df.drop_duplicates(subset=subset)
        else:
            df_clean = df.drop_duplicates()
        
        removed_count = len(df) - len(df_clean)
        
        if output_file is None:
            output_file = input_file
        
        df_clean.to_csv(output_file, index=False)
        
        print(f"Removed {removed_count} duplicate rows")
        print(f"Original rows: {len(df)}")
        print(f"Cleaned rows: {len(df_clean)}")
        print(f"Saved to: {output_file}")
        
        return df_clean
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file]")
        print("Example: python data_cleaner.py data.csv cleaned_data.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_duplicates(input_file, output_file)
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_columns = df.columns.tolist()
    
    def remove_outliers_iqr(self, column, multiplier=1.5):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        return self
    
    def remove_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[column]))
        self.df = self.df[z_scores < threshold]
        return self
    
    def normalize_minmax(self, column):
        min_val = self.df[column].min()
        max_val = self.df[column].max()
        self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        return self
    
    def normalize_zscore(self, column):
        mean_val = self.df[column].mean()
        std_val = self.df[column].std()
        self.df[column] = (self.df[column] - mean_val) / std_val
        return self
    
    def fill_missing_mean(self, column):
        self.df[column] = self.df[column].fillna(self.df[column].mean())
        return self
    
    def fill_missing_median(self, column):
        self.df[column] = self.df[column].fillna(self.df[column].median())
        return self
    
    def drop_duplicates(self):
        self.df = self.df.drop_duplicates()
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def summary(self):
        print(f"Original shape: {len(self.df)} rows, {len(self.df.columns)} columns")
        print(f"Missing values per column:")
        for col in self.df.columns:
            missing = self.df[col].isnull().sum()
            if missing > 0:
                print(f"  {col}: {missing} missing values")
        print(f"Data types:")
        for col in self.df.columns:
            dtype = self.df[col].dtype
            print(f"  {col}: {dtype}")

def example_usage():
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 100),
        'feature2': np.random.exponential(50, 100),
        'feature3': np.random.randint(1, 100, 100)
    }
    df = pd.DataFrame(data)
    
    cleaner = DataCleaner(df)
    cleaner.remove_outliers_iqr('feature1') \
           .normalize_minmax('feature2') \
           .drop_duplicates()
    
    cleaned_df = cleaner.get_cleaned_data()
    cleaner.summary()
    
    return cleaned_df

if __name__ == "__main__":
    result = example_usage()
    print(f"Cleaned data shape: {result.shape}")