import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None):
    """
    Load a CSV file, perform basic cleaning operations,
    and optionally save the cleaned data.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Original shape: {df.shape}")
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        print(f"After removing duplicates: {df.shape}")
        
        # Drop columns with all NaN values
        df = df.dropna(axis=1, how='all')
        print(f"After dropping all-NaN columns: {df.shape}")
        
        # Fill numeric column NaN values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"Filled NaN in {col} with median: {median_val}")
        
        # Fill categorical column NaN values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isna().any():
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_val)
                print(f"Filled NaN in {col} with mode: {mode_val}")
        
        # Remove rows where more than 50% of values are NaN
        threshold = len(df.columns) * 0.5
        df = df.dropna(thresh=threshold)
        print(f"After dropping rows with >50% NaN: {df.shape}")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The file is empty")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate the structure and content of a DataFrame.
    """
    if df is None or df.empty:
        print("DataFrame is empty or None")
        return False
    
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {list(df.columns)}")
    print(f"DataFrame dtypes:\n{df.dtypes}")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    # Check for any remaining NaN values
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"Warning: DataFrame contains {nan_count} NaN values")
        nan_by_col = df.isna().sum()
        print("NaN values by column:")
        print(nan_by_col[nan_by_col > 0])
    
    return True

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, output_file)
    
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df)
        if is_valid:
            print("Data cleaning completed successfully")
        else:
            print("Data validation failed")import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, fill_strategy='mean', drop_threshold=0.5):
    """
    Clean CSV data by handling missing values and removing invalid rows.
    
    Parameters:
    file_path (str): Path to input CSV file
    output_path (str): Path for cleaned output CSV (optional)
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero')
    drop_threshold (float): Threshold for dropping rows with too many missing values (0-1)
    
    Returns:
    pandas.DataFrame: Cleaned dataframe
    """
    
    # Read CSV file
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")
    
    # Calculate missing value percentage per row
    missing_per_row = df.isnull().sum(axis=1) / df.shape[1]
    
    # Drop rows with too many missing values
    rows_before = len(df)
    df = df[missing_per_row < drop_threshold]
    rows_dropped = rows_before - len(df)
    
    # Fill missing values based on strategy
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if fill_strategy == 'mean':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
    elif fill_strategy == 'median':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
    elif fill_strategy == 'mode':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
    elif fill_strategy == 'zero':
        df = df.fillna(0)
    else:
        raise ValueError(f"Unknown fill strategy: {fill_strategy}")
    
    # For non-numeric columns, fill with most frequent value
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Remove duplicate rows
    duplicates_removed = df.duplicated().sum()
    df = df.drop_duplicates()
    
    # Save cleaned data if output path provided
    if output_path:
        df.to_csv(output_path, index=False)
    
    # Print cleaning summary
    print(f"Data cleaning completed:")
    print(f"  - Rows dropped due to missing values: {rows_dropped}")
    print(f"  - Duplicate rows removed: {duplicates_removed}")
    print(f"  - Missing values filled using: {fill_strategy} strategy")
    print(f"  - Final dataset shape: {df.shape}")
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pandas.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    
    if inf_count > 0:
        print(f"Warning: Found {inf_count} infinite values in numeric columns")
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data for demonstration
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6],
        'temperature': [25.5, np.nan, 22.0, np.nan, 30.2, 25.5],
        'humidity': [60, 65, np.nan, 70, 75, 60],
        'pressure': [1013, 1012, 1015, np.nan, np.nan, 1013],
        'location': ['A', 'B', 'A', 'B', 'A', 'A']
    }
    
    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv('sample_data.csv', index=False)
    
    # Clean the sample data
    cleaned_df = clean_csv_data(
        'sample_data.csv',
        output_path='cleaned_data.csv',
        fill_strategy='mean',
        drop_threshold=0.6
    )
    
    # Validate the cleaned data
    is_valid = validate_dataframe(cleaned_df, required_columns=['id', 'temperature'])
    
    if is_valid:
        print("Data validation passed")
    else:
        print("Data validation failed")