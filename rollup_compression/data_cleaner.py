import pandas as pd
import numpy as np
from scipy import stats

def load_data(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    """Remove outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def remove_outliers_zscore(df, column, threshold=3):
    """Remove outliers using Z-score method."""
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def normalize_minmax(df, column):
    """Normalize column using Min-Max scaling."""
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def normalize_zscore(df, column):
    """Normalize column using Z-score normalization."""
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column] = (df[column] - mean_val) / std_val
    return df

def clean_dataset(input_file, output_file, outlier_method='iqr', normalize_method='minmax'):
    """Main function to clean dataset."""
    df = load_data(input_file)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if outlier_method == 'iqr':
            df = remove_outliers_iqr(df, col)
        elif outlier_method == 'zscore':
            df = remove_outliers_zscore(df, col)
    
    for col in numeric_cols:
        if normalize_method == 'minmax':
            df = normalize_minmax(df, col)
        elif normalize_method == 'zscore':
            df = normalize_zscore(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    return df

if __name__ == "__main__":
    cleaned_df = clean_dataset('raw_data.csv', 'cleaned_data.csv')import pandas as pd
import numpy as np
from scipy import stats

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

def remove_outliers_zscore(data, column, threshold=3):
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    cleaned_df = cleaned_df.dropna()
    return cleaned_df

def validate_data(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True

def sample_data_cleaning():
    np.random.seed(42)
    data = {
        'id': range(100),
        'feature_a': np.random.normal(50, 15, 100),
        'feature_b': np.random.exponential(10, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    df = pd.DataFrame(data)
    df.loc[10, 'feature_a'] = 200
    df.loc[20, 'feature_b'] = 150
    
    numeric_cols = ['feature_a', 'feature_b']
    cleaned_df = clean_dataset(df, numeric_cols)
    
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Outliers detected in feature_a: {len(detect_outliers_iqr(df, 'feature_a'))}")
    
    return cleaned_df

if __name__ == "__main__":
    result_df = sample_data_cleaning()
    print(result_df.head())