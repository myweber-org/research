
import pandas as pd
import numpy as np
from datetime import datetime

def clean_dataframe(df, date_column='date', id_column='id'):
    """
    Clean a DataFrame by removing duplicates and standardizing date formats.
    """
    if df.empty:
        return df
    
    cleaned_df = df.copy()
    
    if id_column in cleaned_df.columns:
        cleaned_df = cleaned_df.drop_duplicates(subset=[id_column], keep='first')
    
    if date_column in cleaned_df.columns:
        cleaned_df[date_column] = pd.to_datetime(cleaned_df[date_column], errors='coerce')
        cleaned_df[date_column] = cleaned_df[date_column].dt.strftime('%Y-%m-%d')
    
    cleaned_df = cleaned_df.replace([np.inf, -np.inf], np.nan)
    cleaned_df = cleaned_df.dropna(how='all')
    
    return cleaned_df

def validate_data(df, required_columns):
    """
    Validate that required columns exist in the DataFrame.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

def process_data_file(input_path, output_path, date_column='date', id_column='id'):
    """
    Process a CSV file: clean data and save to output path.
    """
    try:
        df = pd.read_csv(input_path)
        
        required_cols = [date_column, id_column]
        validate_data(df, required_cols)
        
        cleaned_df = clean_dataframe(df, date_column, id_column)
        
        cleaned_df.to_csv(output_path, index=False)
        print(f"Data cleaned successfully. Saved to {output_path}")
        
        return cleaned_df
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    result = process_data_file(input_file, output_file)
    
    if result is not None:
        print(f"Processed {len(result)} records")
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataset(file_path, output_path):
    data = pd.read_csv(file_path)
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        data = remove_outliers_iqr(data, col)
    
    data.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    print(f"Original rows: {len(pd.read_csv(file_path))}, Cleaned rows: {len(data)}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    clean_dataset(input_file, output_file)