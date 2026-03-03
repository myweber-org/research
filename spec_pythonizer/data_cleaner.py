
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def clean_dataset(df, columns_to_clean=None):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns_to_clean is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns_to_clean = numeric_cols
    
    cleaned_df = df.copy()
    
    for column in columns_to_clean:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[::100, 'A'] = 500
    
    print("Original dataset shape:", df.shape)
    print("\nOriginal statistics for column 'A':")
    print(calculate_summary_statistics(df, 'A'))
    
    cleaned_df = clean_dataset(df, ['A'])
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("\nCleaned statistics for column 'A':")
    print(calculate_summary_statistics(cleaned_df, 'A'))
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None):
    """
    Load a CSV file, clean missing values, and save cleaned data.
    """
    try:
        df = pd.read_csv(input_path)
        print(f"Original shape: {df.shape}")
        
        # Remove duplicate rows
        df.drop_duplicates(inplace=True)
        
        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        # Remove columns with more than 50% missing values
        threshold = len(df) * 0.5
        df = df.loc[:, df.isnull().sum() < threshold]
        
        print(f"Cleaned shape: {df.shape}")
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return df
    
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return None
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return None

def validate_dataframe(df):
    """
    Perform basic validation on cleaned dataframe.
    """
    if df is None:
        return False
    
    checks = {
        'has_data': len(df) > 0,
        'has_columns': len(df.columns) > 0,
        'no_duplicates': len(df) == len(df.drop_duplicates()),
        'missing_values': df.isnull().sum().sum() == 0
    }
    
    for check_name, result in checks.items():
        print(f"{check_name}: {'PASS' if result else 'FAIL'}")
    
    return all(checks.values())

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, output_file)
    
    if cleaned_df is not None:
        validation_passed = validate_dataframe(cleaned_df)
        if validation_passed:
            print("Data cleaning completed successfully.")
        else:
            print("Data cleaning completed with warnings.")