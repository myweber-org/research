
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range (IQR) method.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.

    Returns:
    pd.DataFrame: DataFrame with outliers removed from the specified column.
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

def example_usage():
    np.random.seed(42)
    data = {
        'id': range(100),
        'value': np.concatenate([
            np.random.normal(50, 5, 90),
            np.random.normal(150, 10, 10)
        ])
    }
    df = pd.DataFrame(data)
    print(f"Original shape: {df.shape}")
    print(f"Original stats:\n{df['value'].describe()}")

    cleaned_df = remove_outliers_iqr(df, 'value')
    print(f"\nCleaned shape: {cleaned_df.shape}")
    print(f"Cleaned stats:\n{cleaned_df['value'].describe()}")

if __name__ == "__main__":
    example_usage()