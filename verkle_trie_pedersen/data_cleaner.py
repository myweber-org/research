
import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Remove outliers using z-score method
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    
    # Normalize numeric columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
    
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = load_and_clean_data(input_file)
    save_cleaned_data(cleaned_df, output_file)import pandas as pd
import numpy as np

def clean_dataset(df, duplicate_threshold=0.8, missing_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    duplicate_threshold (float): Threshold for considering rows as duplicates (0.0 to 1.0)
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    original_shape = df.shape
    print(f"Original dataset shape: {original_shape}")
    
    # Remove exact duplicates
    df_cleaned = df.drop_duplicates()
    print(f"Removed {original_shape[0] - df_cleaned.shape[0]} exact duplicate rows")
    
    # Remove approximate duplicates based on similarity threshold
    if duplicate_threshold < 1.0:
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            similarity_matrix = df_cleaned[numeric_cols].corr().abs()
            duplicate_mask = similarity_matrix.mean(axis=1) > duplicate_threshold
            df_cleaned = df_cleaned[~duplicate_mask]
            print(f"Removed {duplicate_mask.sum()} approximate duplicate rows")
    
    # Handle missing values
    missing_count = df_cleaned.isnull().sum().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values")
        
        if missing_strategy == 'drop':
            df_cleaned = df_cleaned.dropna()
            print("Removed rows with missing values")
        elif missing_strategy == 'mean':
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_cleaned[col].isnull().any():
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
            print("Filled missing numeric values with column mean")
        elif missing_strategy == 'median':
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_cleaned[col].isnull().any():
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
            print("Filled missing numeric values with column median")
        elif missing_strategy == 'mode':
            for col in df_cleaned.columns:
                if df_cleaned[col].isnull().any():
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
            print("Filled missing values with column mode")
    
    final_shape = df_cleaned.shape
    print(f"Cleaned dataset shape: {final_shape}")
    print(f"Removed {original_shape[0] - final_shape[0]} rows total")
    
    return df_cleaned

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if DataFrame is valid
    """
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

def export_cleaned_data(df, output_path, format='csv'):
    """
    Export cleaned DataFrame to file.
    
    Parameters:
    df (pd.DataFrame): Cleaned DataFrame
    output_path (str): Path to save the file
    format (str): Output format ('csv', 'excel', 'json')
    """
    
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'excel':
        df.to_excel(output_path, index=False)
    elif format == 'json':
        df.to_json(output_path, orient='records')
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Cleaned data exported to {output_path}")

# Example usage
if __name__ == "__main__":
    # Create sample data with duplicates and missing values
    sample_data = {
        'id': [1, 2, 3, 4, 5, 1, 2],
        'value_a': [10.5, 20.3, 15.7, None, 18.9, 10.5, 20.3],
        'value_b': [100, 200, 150, 250, 180, 100, 200],
        'category': ['A', 'B', 'A', 'C', 'B', 'A', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    
    try:
        validate_dataframe(df, required_columns=['id', 'value_a', 'value_b'])
        cleaned_df = clean_dataset(df, duplicate_threshold=0.9, missing_strategy='mean')
        export_cleaned_data(cleaned_df, 'cleaned_data.csv', format='csv')
    except Exception as e:
        print(f"Error during data cleaning: {e}")