import pandas as pd
import numpy as np

def clean_csv_data(input_path, output_path, missing_strategy='mean'):
    """
    Load a CSV file, clean missing values, and save the cleaned data.
    
    Parameters:
    input_path (str): Path to the input CSV file.
    output_path (str): Path to save the cleaned CSV file.
    missing_strategy (str): Strategy for handling missing values.
                            Options: 'mean', 'median', 'drop', 'zero'.
    """
    try:
        df = pd.read_csv(input_path)
        print(f"Original data shape: {df.shape}")
        
        if missing_strategy == 'drop':
            df_cleaned = df.dropna()
        elif missing_strategy == 'zero':
            df_cleaned = df.fillna(0)
        elif missing_strategy == 'mean':
            df_cleaned = df.fillna(df.mean(numeric_only=True))
        elif missing_strategy == 'median':
            df_cleaned = df.fillna(df.median(numeric_only=True))
        else:
            raise ValueError(f"Unknown strategy: {missing_strategy}")
        
        print(f"Cleaned data shape: {df_cleaned.shape}")
        print(f"Missing values after cleaning: {df_cleaned.isnull().sum().sum()}")
        
        df_cleaned.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def detect_outliers_iqr(df, column):
    """
    Detect outliers in a DataFrame column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to check for outliers.
    
    Returns:
    pd.DataFrame: Rows identified as outliers.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, np.nan, 30, 40, 50, 60],
        'C': ['x', 'y', 'z', 'x', 'y', 'z']
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned = clean_csv_data('sample_data.csv', 'cleaned_sample.csv', 'mean')
    
    if cleaned is not None:
        outliers = detect_outliers_iqr(cleaned, 'A')
        print(f"Outliers in column 'A': {len(outliers)} rows")