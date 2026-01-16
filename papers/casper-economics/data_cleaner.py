
def remove_duplicates_preserve_order(seq):
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    
    if drop_duplicates:
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")
    
    if fill_missing:
        for column in df.columns:
            if df[column].isnull().any():
                if fill_missing == 'mean' and pd.api.types.is_numeric_dtype(df[column]):
                    df[column] = df[column].fillna(df[column].mean())
                elif fill_missing == 'median' and pd.api.types.is_numeric_dtype(df[column]):
                    df[column] = df[column].fillna(df[column].median())
                elif fill_missing == 'mode':
                    df[column] = df[column].fillna(df[column].mode()[0])
                elif fill_missing == 'ffill':
                    df[column] = df[column].fillna(method='ffill')
                elif fill_missing == 'bfill':
                    df[column] = df[column].fillna(method='bfill')
                else:
                    df[column] = df[column].fillna(0)
    
    print(f"Dataset cleaned: {original_shape} -> {df.shape}")
    return df

def validate_dataset(df, required_columns=None):
    """
    Validate dataset structure and content.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    validation_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    return validation_report

def main():
    # Example usage
    data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10, 20, 20, np.nan, 40, 50],
        'category': ['A', 'B', 'B', 'C', None, 'A']
    }
    
    df = pd.DataFrame(data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50)
    
    # Clean the dataset
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    # Validate the dataset
    report = validate_dataset(cleaned_df, required_columns=['id', 'value'])
    print("\nValidation report:")
    for key, value in report.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()