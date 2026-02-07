import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean', output_path=None):
    """
    Load a CSV file, clean missing values, and optionally save cleaned data.
    
    Args:
        filepath (str): Path to input CSV file.
        fill_strategy (str): Strategy for filling missing values.
                             Options: 'mean', 'median', 'mode', 'zero', 'drop'.
        output_path (str, optional): Path to save cleaned CSV. If None, returns DataFrame.
    
    Returns:
        pd.DataFrame or None: Cleaned DataFrame if output_path is None, else None.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded data with shape: {df.shape}")
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values.")
            
            if fill_strategy == 'drop':
                df_cleaned = df.dropna()
                print(f"Dropped rows with missing values. New shape: {df_cleaned.shape}")
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    if df[col].isnull().any():
                        if fill_strategy == 'mean':
                            fill_value = df[col].mean()
                        elif fill_strategy == 'median':
                            fill_value = df[col].median()
                        elif fill_strategy == 'mode':
                            fill_value = df[col].mode()[0]
                        elif fill_strategy == 'zero':
                            fill_value = 0
                        else:
                            raise ValueError(f"Unknown fill strategy: {fill_strategy}")
                        
                        df[col].fillna(fill_value, inplace=True)
                        print(f"Filled missing values in column '{col}' with {fill_strategy}: {fill_value}")
                
                object_cols = df.select_dtypes(include=['object']).columns
                for col in object_cols:
                    if df[col].isnull().any():
                        df[col].fillna('Unknown', inplace=True)
                        print(f"Filled missing values in column '{col}' with 'Unknown'")
            
            df_cleaned = df
        else:
            print("No missing values found.")
            df_cleaned = df
        
        if output_path:
            df_cleaned.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
            return None
        else:
            return df_cleaned
            
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def summarize_data(df):
    """
    Generate summary statistics for a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        dict: Dictionary containing summary statistics.
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_summary': df.describe().to_dict() if df.select_dtypes(include=[np.number]).shape[1] > 0 else {}
    }
    return summary

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [5.1, np.nan, 7.3, 8.4, 9.5],
        'C': ['X', 'Y', 'Z', np.nan, 'W'],
        'D': [10, 20, 30, 40, 50]
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', fill_strategy='median')
    
    if cleaned_df is not None:
        summary = summarize_data(cleaned_df)
        print(f"Data shape: {summary['shape']}")
        print(f"Missing values after cleaning: {sum(cleaned_df.isnull().sum().values())}")