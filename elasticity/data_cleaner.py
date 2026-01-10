
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    
    Args:
        input_list (list): List potentially containing duplicates.
    
    Returns:
        list: List with duplicates removed.
    """
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_data(data):
    """
    Clean data by removing duplicates and None values.
    
    Args:
        data (list): Raw data list.
    
    Returns:
        list: Cleaned data list.
    """
    if not isinstance(data, list):
        raise TypeError("Input must be a list")
    
    # Remove None values first
    filtered = [item for item in data if item is not None]
    
    # Remove duplicates
    cleaned = remove_duplicates(filtered)
    
    return cleaned

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, None, 4, 3, 5, None, 1]
    print(f"Original data: {sample_data}")
    print(f"Cleaned data: {clean_data(sample_data)}")
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataset(input_path, output_path):
    try:
        df = pd.read_csv(input_path)
        print(f"Original dataset shape: {df.shape}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            original_len = len(df)
            df = remove_outliers_iqr(df, col)
            removed = original_len - len(df)
            if removed > 0:
                print(f"Removed {removed} outliers from column '{col}'")
        
        df.to_csv(output_path, index=False)
        print(f"Cleaned dataset saved to {output_path}")
        print(f"Final dataset shape: {df.shape}")
        
        return True
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return False

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    clean_dataset(input_file, output_file)