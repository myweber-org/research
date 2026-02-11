import numpy as np
import pandas as pd

def remove_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

def normalize_minmax(dataframe, column):
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    if max_val == min_val:
        return dataframe[column].apply(lambda x: 0.0)
    return dataframe[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def standardize_zscore(dataframe, column):
    mean_val = dataframe[column].mean()
    std_val = dataframe[column].std()
    if std_val == 0:
        return dataframe[column].apply(lambda x: 0.0)
    return dataframe[column].apply(lambda x: (x - mean_val) / std_val)

def handle_missing_values(dataframe, strategy='mean'):
    df_copy = dataframe.copy()
    for col in df_copy.columns:
        if df_copy[col].isnull().any():
            if strategy == 'mean' and pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
            elif strategy == 'mode':
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
            else:
                df_copy[col].fillna(0, inplace=True)
    return df_copy

def get_data_summary(dataframe):
    summary = {
        'shape': dataframe.shape,
        'columns': list(dataframe.columns),
        'dtypes': dataframe.dtypes.to_dict(),
        'missing_values': dataframe.isnull().sum().to_dict(),
        'numeric_stats': {}
    }
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': dataframe[col].mean(),
            'std': dataframe[col].std(),
            'min': dataframe[col].min(),
            'max': dataframe[col].max(),
            'median': dataframe[col].median()
        }
    return summary