import pandas as pd

def clean_dataset(df, sort_column=None):
    """
    Cleans a pandas DataFrame by removing duplicate rows and optionally sorting.

    Args:
        df (pd.DataFrame): The input DataFrame to clean.
        sort_column (str, optional): The column name to sort by. Defaults to None.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # Remove duplicate rows, keeping the first occurrence
    cleaned_df = df.drop_duplicates()

    # Sort the DataFrame if a column name is provided
    if sort_column and sort_column in cleaned_df.columns:
        cleaned_df = cleaned_df.sort_values(by=sort_column).reset_index(drop=True)
    else:
        cleaned_df = cleaned_df.reset_index(drop=True)

    return cleaned_df

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = {'A': [3, 1, 2, 1, 3], 'B': ['x', 'y', 'z', 'y', 'x']}
#     df = pd.DataFrame(sample_data)
#     print("Original DataFrame:")
#     print(df)
#     cleaned = clean_dataset(df, sort_column='A')
#     print("\nCleaned DataFrame (sorted by column 'A'):")
#     print(cleaned)