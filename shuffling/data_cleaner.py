
import pandas as pd
import numpy as np
from scipy import stats

def load_data(filepath):
    return pd.read_csv(filepath)

def remove_outliers(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def normalize_column(df, column):
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df

def clean_data(input_file, output_file):
    df = load_data(input_file)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers(df, col)
        df = normalize_column(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")

if __name__ == "__main__":
    clean_data("raw_data.csv", "cleaned_data.csv")
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    cleaned_df = cleaned_df.dropna()
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

def validate_data(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return Truedef remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    Returns a new list with unique elements.
    """
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_data_with_threshold(data, threshold=None):
    """
    Clean data by removing duplicates and optionally filtering by threshold.
    If threshold is provided, only items with count >= threshold are kept.
    """
    from collections import Counter
    
    if not data:
        return []
    
    # Remove duplicates while preserving order
    unique_data = remove_duplicates(data)
    
    if threshold is not None:
        counter = Counter(data)
        filtered_data = [item for item in unique_data if counter[item] >= threshold]
        return filtered_data
    
    return unique_data

def validate_data(data, validator_func=None):
    """
    Validate data using optional validator function.
    If no validator provided, checks for non-None values.
    """
    if validator_func is None:
        validator_func = lambda x: x is not None
    
    return [item for item in data if validator_func(item)]

# Example usage functions
def example_usage():
    sample_data = [1, 2, 2, 3, 4, 4, 4, 5, None, 6, 6]
    
    print("Original data:", sample_data)
    print("Without duplicates:", remove_duplicates(sample_data))
    print("Threshold 2:", clean_data_with_threshold(sample_data, threshold=2))
    print("Validated data:", validate_data(sample_data))
    
    # Custom validator example
    def is_even(x):
        return x is not None and x % 2 == 0
    
    print("Even numbers only:", validate_data(sample_data, is_even))

if __name__ == "__main__":
    example_usage()