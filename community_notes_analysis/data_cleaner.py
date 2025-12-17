import pandas as pd
import numpy as np
from scipy import stats

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

def impute_missing_values(data, column, method='mean'):
    if method == 'mean':
        fill_value = data[column].mean()
    elif method == 'median':
        fill_value = data[column].median()
    elif method == 'mode':
        fill_value = data[column].mode()[0]
    else:
        fill_value = 0
    
    data[column] = data[column].fillna(fill_value)
    return data

def remove_duplicates(data, subset=None):
    if subset:
        return data.drop_duplicates(subset=subset)
    return data.drop_duplicates()

def standardize_column(data, column):
    mean = data[column].mean()
    std = data[column].std()
    data[column] = (data[column] - mean) / std
    return data

def clean_dataset(data, config):
    cleaned_data = data.copy()
    
    for column in config.get('outlier_columns', []):
        outliers = detect_outliers_iqr(cleaned_data, column)
        if not outliers.empty:
            cleaned_data = cleaned_data.drop(outliers.index)
    
    for column, method in config.get('impute_columns', {}).items():
        cleaned_data = impute_missing_values(cleaned_data, column, method)
    
    if config.get('remove_duplicates', False):
        cleaned_data = remove_duplicates(cleaned_data, config.get('duplicate_subset'))
    
    for column in config.get('standardize_columns', []):
        cleaned_data = standardize_column(cleaned_data, column)
    
    return cleaned_data

def main():
    sample_data = pd.DataFrame({
        'A': [1, 2, 3, 100, 5, np.nan, 7, 8, 9, 10],
        'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'C': ['x', 'y', 'x', 'y', 'x', 'y', 'x', 'y', 'x', 'y']
    })
    
    config = {
        'outlier_columns': ['A'],
        'impute_columns': {'A': 'mean'},
        'remove_duplicates': True,
        'duplicate_subset': ['C'],
        'standardize_columns': ['B']
    }
    
    cleaned = clean_dataset(sample_data, config)
    print("Original data shape:", sample_data.shape)
    print("Cleaned data shape:", cleaned.shape)
    print("\nCleaned data:")
    print(cleaned)

if __name__ == "__main__":
    main()import pandas as pd
import numpy as np

def clean_data(input_file, output_file):
    """
    Clean a CSV file by removing duplicates, handling missing values,
    and standardizing column names.
    """
    try:
        df = pd.read_csv(input_file)
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Handle missing values for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # Handle missing values for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna('unknown')
        
        # Remove rows where critical columns are null
        critical_columns = ['id', 'timestamp']
        existing_critical = [col for col in critical_columns if col in df.columns]
        if existing_critical:
            df = df.dropna(subset=existing_critical)
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Data cleaned successfully. Output saved to {output_file}")
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except pd.errors.EmptyDataError:
        print("Error: Input file is empty.")
        return False
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return False

if __name__ == "__main__":
    # Example usage
    clean_data('raw_data.csv', 'cleaned_data.csv')
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = self.df.select_dtypes(exclude=[np.number]).columns

    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean':
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].mean())
        elif strategy == 'median':
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].median())
        elif strategy == 'mode':
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].mode().iloc[0])
        elif strategy == 'constant' and fill_value is not None:
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(fill_value)
        else:
            raise ValueError("Invalid strategy or missing fill_value for constant strategy")
        
        self.df[self.categorical_columns] = self.df[self.categorical_columns].fillna('Unknown')
        return self

    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        return self

    def normalize_data(self, method='minmax'):
        if method == 'minmax':
            for col in self.numeric_columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            for col in self.numeric_columns:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
        else:
            raise ValueError("Invalid normalization method")
        return self

    def get_cleaned_data(self):
        return self.df

def clean_dataset(df, missing_strategy='mean', outlier_removal=True, normalization=None):
    cleaner = DataCleaner(df)
    cleaner.handle_missing_values(strategy=missing_strategy)
    
    if outlier_removal:
        cleaner.remove_outliers_iqr()
    
    if normalization:
        cleaner.normalize_data(method=normalization)
    
    return cleaner.get_cleaned_data()