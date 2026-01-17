
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
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if fill_missing == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif fill_missing == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif fill_missing == 'zero':
            df[numeric_cols] = df[numeric_cols].fillna(0)
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Warning: {missing_count} missing values remain in non-numeric columns")
    
    print(f"Dataset cleaned: {original_shape} -> {df.shape}")
    return df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    """
    if len(df) < min_rows:
        raise ValueError(f"Dataset must have at least {min_rows} rows")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

def main():
    # Example usage
    data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, np.nan, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'C', None, 'A']
    }
    
    df = pd.DataFrame(data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    try:
        validate_data(cleaned_df, required_columns=['id', 'value'], min_rows=3)
        print("\nData validation passed")
    except ValueError as e:
        print(f"\nData validation failed: {e}")

if __name__ == "__main__":
    main()import numpy as np
import pandas as pd

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
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
    
    def fill_missing_numeric(self, column, strategy='mean'):
        if column not in self.numeric_columns:
            raise ValueError(f"Column {column} is not numeric")
        
        if strategy == 'mean':
            fill_value = self.df[column].mean()
        elif strategy == 'median':
            fill_value = self.df[column].median()
        elif strategy == 'mode':
            fill_value = self.df[column].mode().iloc[0] if not self.df[column].mode().empty else 0
        else:
            raise ValueError("Strategy must be 'mean', 'median', or 'mode'")
        
        self.df[column].fillna(fill_value, inplace=True)
        return self
    
    def encode_categorical(self, column, method='onehot'):
        if column not in self.categorical_columns:
            raise ValueError(f"Column {column} is not categorical")
        
        if method == 'onehot':
            dummies = pd.get_dummies(self.df[column], prefix=column, drop_first=True)
            self.df = pd.concat([self.df.drop(column, axis=1), dummies], axis=1)
            self.categorical_columns.remove(column)
            self.numeric_columns.extend(dummies.columns.tolist())
        elif method == 'label':
            unique_vals = self.df[column].unique()
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            self.df[column] = self.df[column].map(mapping)
            self.categorical_columns.remove(column)
            self.numeric_columns.append(column)
        else:
            raise ValueError("Method must be 'onehot' or 'label'")
        
        return self
    
    def get_cleaned_data(self):
        return self.df.copy()
    
    def summary(self):
        print(f"Data shape: {self.df.shape}")
        print(f"Numeric columns: {len(self.numeric_columns)}")
        print(f"Categorical columns: {len(self.categorical_columns)}")
        print("\nMissing values:")
        print(self.df.isnull().sum())
        print("\nData types:")
        print(self.df.dtypes)

def example_usage():
    data = {
        'age': [25, 30, 35, 200, 28, 32, 150, 29, np.nan, 31],
        'salary': [50000, 60000, 70000, 80000, 55000, 65000, 75000, 58000, 62000, 1000000],
        'department': ['IT', 'HR', 'IT', 'Finance', 'HR', 'IT', 'Finance', 'IT', 'HR', 'IT'],
        'experience': [2, 5, 8, 12, 3, 6, 10, 4, 7, 15]
    }
    
    df = pd.DataFrame(data)
    cleaner = DataCleaner(df)
    
    cleaner.summary()
    
    cleaner.remove_outliers_iqr('age')
    cleaner.remove_outliers_iqr('salary')
    cleaner.fill_missing_numeric('age', strategy='mean')
    cleaner.normalize_column('experience', method='minmax')
    cleaner.encode_categorical('department', method='onehot')
    
    cleaned_df = cleaner.get_cleaned_data()
    print("\nCleaned data shape:", cleaned_df.shape)
    print("\nCleaned data head:")
    print(cleaned_df.head())
    
    return cleaned_df

if __name__ == "__main__":
    example_usage()