
import numpy as np

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    cleaned_df = df.copy()

    if drop_duplicates:
        initial_rows = cleaned_df.shape[0]
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - cleaned_df.shape[0]
        print(f"Removed {removed} duplicate rows.")

    if cleaned_df.isnull().sum().any():
        print("Handling missing values...")
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        categorical_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns

        if fill_missing == 'mean':
            for col in numeric_cols:
                if cleaned_df[col].isnull().any():
                    cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
        elif fill_missing == 'median':
            for col in numeric_cols:
                if cleaned_df[col].isnull().any():
                    cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
        elif fill_missing == 'mode':
            for col in numeric_cols:
                if cleaned_df[col].isnull().any():
                    cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
        else:
            cleaned_df.fillna(fill_missing, inplace=True)

        for col in categorical_cols:
            if cleaned_df[col].isnull().any():
                cleaned_df[col].fillna('Unknown', inplace=True)

    print(f"Data cleaning complete. Final shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate the DataFrame structure and content.
    """
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    if df.empty:
        raise ValueError("DataFrame is empty")

    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, None, 5],
        'B': [10.5, None, 10.5, 15.2, 20.1, 20.1],
        'C': ['x', 'y', 'x', 'z', None, 'y'],
        'D': [100, 200, 100, 300, 400, 500]
    }

    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")

    cleaned = clean_dataframe(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)

    try:
        validate_dataframe(cleaned, required_columns=['A', 'B', 'C', 'D'])
        print("\nData validation passed.")
    except ValueError as e:
        print(f"\nData validation failed: {e}")