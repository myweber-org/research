import pandas as pd

def clean_dataset(df, subset=None, fill_method='mean'):
    """
    Clean a pandas DataFrame by removing duplicate rows and handling missing values.

    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        subset (list, optional): Column labels to consider for identifying duplicates.
                                 If None, all columns are used.
        fill_method (str): Method to fill missing values.
                           Options: 'mean', 'median', 'mode', or 'drop'.
                           Default is 'mean'.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()

    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates(subset=subset, keep='first')

    # Handle missing values
    if fill_method == 'drop':
        cleaned_df = cleaned_df.dropna()
    else:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_method == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif fill_method == 'median':
                    fill_value = cleaned_df[col].median()
                elif fill_method == 'mode':
                    fill_value = cleaned_df[col].mode()[0]
                else:
                    raise ValueError(f"Unsupported fill_method: {fill_method}")
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)

        # For non-numeric columns, fill with mode or drop if mode not applicable
        non_numeric_cols = cleaned_df.select_dtypes(exclude=['number']).columns
        for col in non_numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_method == 'drop':
                    cleaned_df = cleaned_df.dropna(subset=[col])
                else:
                    # For categorical columns, fill with mode
                    if not cleaned_df[col].mode().empty:
                        fill_value = cleaned_df[col].mode()[0]
                        cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                    else:
                        # If mode is empty (e.g., all NaN), drop the column
                        cleaned_df = cleaned_df.drop(columns=[col])

    return cleaned_df