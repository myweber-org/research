
import numpy as np
import pandas as pd

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        
    def remove_outliers_iqr(self, column, multiplier=1.5):
        if column not in self.numeric_columns:
            raise ValueError(f"Column {column} is not numeric")
        
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        mask = (self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)
        self.df = self.df[mask]
        return self
    
    def normalize_column(self, column, method='minmax'):
        if column not in self.numeric_columns:
            raise ValueError(f"Column {column} is not numeric")
        
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if max_val != min_val:
                self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            if std_val > 0:
                self.df[column] = (self.df[column] - mean_val) / std_val
        else:
            raise ValueError("Method must be 'minmax' or 'zscore'")
        
        return self
    
    def fill_missing(self, column, strategy='mean'):
        if column not in self.df.columns:
            raise ValueError(f"Column {column} does not exist")
        
        if strategy == 'mean' and column in self.numeric_columns:
            fill_value = self.df[column].mean()
        elif strategy == 'median' and column in self.numeric_columns:
            fill_value = self.df[column].median()
        elif strategy == 'mode':
            fill_value = self.df[column].mode()[0]
        else:
            fill_value = 0
        
        self.df[column].fillna(fill_value, inplace=True)
        return self
    
    def get_cleaned_data(self):
        return self.df.copy()
    
    def summary(self):
        print(f"Data shape: {self.df.shape}")
        print(f"Numeric columns: {list(self.numeric_columns)}")
        print(f"Missing values per column:")
        print(self.df.isnull().sum())
        return self

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    data['feature_a'][np.random.choice(100, 5)] = np.nan
    data['feature_b'][np.random.choice(100, 3)] = np.nan
    return pd.DataFrame(data)

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    cleaned_df = (cleaner
                 .summary()
                 .fill_missing('feature_a', 'mean')
                 .fill_missing('feature_b', 'median')
                 .remove_outliers_iqr('feature_a')
                 .normalize_column('feature_a', 'minmax')
                 .normalize_column('feature_b', 'zscore')
                 .get_cleaned_data())
    
    print("\nCleaned data preview:")
    print(cleaned_df.head())
    print(f"\nFinal shape: {cleaned_df.shape}")