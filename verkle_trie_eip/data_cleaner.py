import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None):
    """
    Load a CSV file, perform basic cleaning operations,
    and optionally save the cleaned data.
    """
    try:
        df = pd.read_csv(file_path)
        
        # Remove duplicate rows
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_rows - len(df)
        
        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Remove columns with more than 50% missing values
        threshold = len(df) * 0.5
        cols_to_drop = [col for col in df.columns if df[col].isnull().sum() > threshold]
        df.drop(columns=cols_to_drop, inplace=True)
        
        # Reset index after cleaning
        df.reset_index(drop=True, inplace=True)
        
        # Print cleaning summary
        print(f"Data cleaning completed:")
        print(f"  - Removed {duplicates_removed} duplicate rows")
        print(f"  - Dropped {len(cols_to_drop)} columns with >50% missing values")
        print(f"  - Final dataset shape: {df.shape}")
        
        # Save cleaned data if output path provided
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for common data quality issues.
    """
    if df is None or df.empty:
        print("Error: DataFrame is empty or None")
        return False
    
    # Check for required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    # Check for completely empty columns
    empty_cols = [col for col in df.columns if df[col].isnull().all()]
    if empty_cols:
        print(f"Warning: Found empty columns: {empty_cols}")
    
    # Check data types consistency
    dtype_summary = df.dtypes.value_counts()
    print("Data type summary:")
    for dtype, count in dtype_summary.items():
        print(f"  {dtype}: {count} columns")
    
    return True

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column using specified method.
    """
    if column not in df.columns:
        print(f"Error: Column '{column}' not found in DataFrame")
        return df
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        initial_count = len(df)
        df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        outliers_removed = initial_count - len(df_clean)
        
        print(f"Removed {outliers_removed} outliers from column '{column}' using IQR method")
        return df_clean
    
    elif method == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        
        initial_count = len(df)
        mask = z_scores < threshold
        df_clean = df.iloc[mask]
        outliers_removed = initial_count - len(df_clean)
        
        print(f"Removed {outliers_removed} outliers from column '{column}' using Z-score method")
        return df_clean
    
    else:
        print(f"Error: Unknown method '{method}'. Use 'iqr' or 'zscore'")
        return df