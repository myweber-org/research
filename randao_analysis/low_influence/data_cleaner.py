
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None):
    """
    Clean CSV data by handling missing values and standardizing formats.
    """
    try:
        df = pd.read_csv(input_path)
        
        # Remove duplicate rows
        initial_count = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_count - len(df)
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical missing values with mode
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Generate output path if not provided
        if output_path is None:
            input_file = Path(input_path)
            output_path = input_file.parent / f"cleaned_{input_file.name}"
        
        # Save cleaned data
        df.to_csv(output_path, index=False)
        
        # Print summary
        print(f"Data cleaning completed:")
        print(f"  - Removed {duplicates_removed} duplicate rows")
        print(f"  - Processed {len(numeric_cols)} numeric columns")
        print(f"  - Processed {len(categorical_cols)} categorical columns")
        print(f"  - Cleaned data saved to: {output_path}")
        
        return df, output_path
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return None, None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None, None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    """
    if df is None or df.empty:
        return False, "DataFrame is empty or None"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    # Check for remaining null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        return False, f"Found null values in columns: {null_counts[null_counts > 0].to_dict()}"
    
    return True, "Data validation passed"

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Name': ['Alice', 'Bob', 'Charlie', None, 'Eve'],
        'Age': [25, None, 30, 35, 40],
        'Score': [85.5, 92.0, None, 78.5, 88.0],
        'Category': ['A', 'B', 'A', None, 'B']
    }
    
    # Create sample CSV
    pd.DataFrame(sample_data).to_csv('sample_data.csv', index=False)
    
    # Clean the data
    cleaned_df, output_file = clean_csv_data('sample_data.csv')
    
    if cleaned_df is not None:
        # Validate cleaned data
        is_valid, message = validate_dataframe(cleaned_df)
        print(f"Validation: {is_valid} - {message}")
        
        # Display cleaned data
        print("\nCleaned Data:")
        print(cleaned_df)