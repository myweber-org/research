
import pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(
    filepath: str,
    missing_strategy: str = 'drop',
    fill_value: Optional[float] = None
) -> pd.DataFrame:
    """
    Load and clean CSV data by handling missing values.
    
    Args:
        filepath: Path to CSV file
        missing_strategy: Strategy for handling missing values
                         ('drop', 'fill', or 'interpolate')
        fill_value: Value to fill missing entries with when strategy is 'fill'
    
    Returns:
        Cleaned DataFrame
    """
    try:
        df = pd.read_csv(filepath)
        
        if missing_strategy == 'drop':
            df_clean = df.dropna()
        elif missing_strategy == 'fill':
            if fill_value is not None:
                df_clean = df.fillna(fill_value)
            else:
                df_clean = df.fillna(df.mean(numeric_only=True))
        elif missing_strategy == 'interpolate':
            df_clean = df.interpolate(method='linear')
        else:
            raise ValueError(f"Unknown strategy: {missing_strategy}")
        
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            df_clean[numeric_cols] = df_clean[numeric_cols].round(4)
        
        return df_clean
        
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        print("Error: CSV file is empty")
        return pd.DataFrame()
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return pd.DataFrame()

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate DataFrame for common data quality issues.
    """
    if df.empty:
        return False
    
    if df.isnull().sum().sum() > 0:
        print("Warning: DataFrame contains missing values")
        return False
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if (df[col] == 0).all():
            print(f"Warning: Column '{col}' contains only zeros")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [5.5, np.nan, 7.7, 8.8, 9.9],
        'C': ['x', 'y', 'z', 'x', 'y']
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned = clean_csv_data('sample_data.csv', missing_strategy='fill')
    print("Cleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_dataframe(cleaned)
    print(f"Data validation result: {is_valid}")