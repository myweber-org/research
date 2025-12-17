
import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_method=None):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Args:
        df: pandas DataFrame to clean
        remove_duplicates: Boolean indicating whether to remove duplicate rows
        fill_method: Method to fill missing values ('mean', 'median', 'mode', or None)
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        
        if fill_method == 'mean':
            for col in numeric_cols:
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
        elif fill_method == 'median':
            for col in numeric_cols:
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
        elif fill_method == 'mode':
            for col in cleaned_df.columns:
                cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None, inplace=True)
    
    # Remove duplicates if specified
    if remove_duplicates:
        cleaned_df.drop_duplicates(inplace=True)
    
    # Reset index after cleaning
    cleaned_df.reset_index(drop=True, inplace=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for basic data quality checks.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of column names that must be present
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'null_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_columns
        validation_results['all_required_columns_present'] = len(missing_columns) == 0
    
    return validation_results

def export_cleaned_data(df, output_path, format='csv'):
    """
    Export cleaned DataFrame to a file.
    
    Args:
        df: pandas DataFrame to export
        output_path: Path where to save the file
        format: File format ('csv', 'excel', 'json')
    """
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'excel':
        df.to_excel(output_path, index=False)
    elif format == 'json':
        df.to_json(output_path, orient='records')
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv', 'excel', or 'json'.")