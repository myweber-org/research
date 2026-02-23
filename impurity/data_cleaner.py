
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def zscore_normalize(df, column):
    mean = df[column].mean()
    std = df[column].std()
    df[column + '_normalized'] = (df[column] - mean) / std
    return df

def minmax_normalize(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def handle_missing_values(df, strategy='mean'):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            df[col].fillna(df[col].mean(), inplace=True)
    elif strategy == 'median':
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
    elif strategy == 'mode':
        for col in numeric_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

def detect_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    outliers = df[z_scores > threshold]
    return outliers

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize_method='zscore', missing_strategy='mean'):
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    
    df_clean = handle_missing_values(df_clean, strategy=missing_strategy)
    
    for col in numeric_columns:
        if outlier_method == 'iqr':
            df_clean = remove_outliers_iqr(df_clean, col)
        elif outlier_method == 'zscore':
            outliers = detect_outliers_zscore(df_clean, col)
            df_clean = df_clean.drop(outliers.index)
    
    for col in numeric_columns:
        if normalize_method == 'zscore':
            df_clean = zscore_normalize(df_clean, col)
        elif normalize_method == 'minmax':
            df_clean = minmax_normalize(df_clean, col)
    
    return df_clean
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_data(df, numeric_columns):
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
        df = normalize_column(df, col)
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_file = 'raw_data.csv'
    output_file = 'cleaned_data.csv'
    numeric_cols = ['age', 'income', 'score']
    
    raw_df = load_dataset(input_file)
    cleaned_df = clean_data(raw_df, numeric_cols)
    save_cleaned_data(cleaned_df, output_file)
    
    print(f"Data cleaning completed. Original rows: {len(raw_df)}, Cleaned rows: {len(cleaned_df)}")