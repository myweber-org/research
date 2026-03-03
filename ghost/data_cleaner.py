
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
    clean_dataset(input_file, output_file)import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas Series using the IQR method.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data to [0, 1] range using min-max scaling.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score normalization (mean=0, std=1).
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    normalized = (data[column] - mean_val) / std_val
    return normalized

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Main cleaning function to process multiple numeric columns.
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize_method == 'minmax':
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df[col] = normalize_zscore(cleaned_df, col)
    
    return cleaned_df.reset_index(drop=True)