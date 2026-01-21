
import pandas as pd
import numpy as np

def clean_csv_data(file_path, strategy='mean', output_path=None):
    """
    Clean a CSV file by handling missing values.
    
    Parameters:
    file_path (str): Path to the input CSV file.
    strategy (str): Strategy for filling missing values. 
                    Options: 'mean', 'median', 'mode', 'drop'.
    output_path (str): Path to save cleaned CSV. If None, returns DataFrame.
    
    Returns:
    pd.DataFrame or None: Cleaned DataFrame if output_path is None.
    """
    try:
        df = pd.read_csv(file_path)
        
        if strategy == 'drop':
            df_cleaned = df.dropna()
        else:
            for column in df.select_dtypes(include=[np.number]).columns:
                if df[column].isnull().any():
                    if strategy == 'mean':
                        fill_value = df[column].mean()
                    elif strategy == 'median':
                        fill_value = df[column].median()
                    elif strategy == 'mode':
                        fill_value = df[column].mode()[0]
                    else:
                        raise ValueError(f"Unknown strategy: {strategy}")
                    
                    df[column] = df[column].fillna(fill_value)
            
            df_cleaned = df
        
        if output_path:
            df_cleaned.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
            return None
        else:
            return df_cleaned
            
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if df.empty:
        print("Warning: DataFrame is empty")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': [1, 2, 3, 4, 5]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned = clean_csv_data('test_data.csv', strategy='mean')
    
    if cleaned is not None:
        print("Original data:")
        print(test_df)
        print("\nCleaned data:")
        print(cleaned)
        
        import os
        os.remove('test_data.csv')
def remove_duplicates_preserve_order(iterable):
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result