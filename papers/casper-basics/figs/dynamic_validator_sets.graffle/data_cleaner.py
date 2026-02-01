import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to process
    factor (float): Multiplier for IQR
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to standardize
    
    Returns:
    pd.Series: Standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    strategy (str): Imputation strategy ('mean', 'median', 'mode', 'constant')
    columns (list): List of columns to process, None for all numeric columns
    
    Returns:
    pd.DataFrame: Dataframe with imputed values
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    result = data.copy()
    
    for col in columns:
        if col not in result.columns:
            continue
            
        if strategy == 'mean':
            fill_value = result[col].mean()
        elif strategy == 'median':
            fill_value = result[col].median()
        elif strategy == 'mode':
            fill_value = result[col].mode()[0] if not result[col].mode().empty else 0
        elif strategy == 'constant':
            fill_value = 0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        result[col] = result[col].fillna(fill_value)
    
    return result

def validate_dataframe(data, required_columns=None, numeric_columns=None):
    """
    Validate dataframe structure and content.
    
    Parameters:
    data (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    numeric_columns (list): List of columns that should be numeric
    
    Returns:
    dict: Dictionary with validation results
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if not isinstance(data, pd.DataFrame):
        validation_result['is_valid'] = False
        validation_result['errors'].append('Input is not a pandas DataFrame')
        return validation_result
    
    if data.empty:
        validation_result['warnings'].append('DataFrame is empty')
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f'Missing required columns: {missing_columns}')
    
    if numeric_columns:
        non_numeric = []
        for col in numeric_columns:
            if col in data.columns and not np.issubdtype(data[col].dtype, np.number):
                non_numeric.append(col)
        
        if non_numeric:
            validation_result['warnings'].append(f'Non-numeric columns marked as numeric: {non_numeric}')
    
    return validation_result

def create_sample_data():
    """
    Create sample data for testing.
    
    Returns:
    pd.DataFrame: Sample dataframe with test data
    """
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'id': range(n_samples),
        'feature_a': np.random.normal(50, 15, n_samples),
        'feature_b': np.random.exponential(2, n_samples),
        'feature_c': np.random.uniform(0, 100, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[np.random.choice(n_samples, 5, replace=False), 'feature_a'] = np.nan
    df.loc[np.random.choice(n_samples, 3, replace=False), 'feature_b'] = np.nan
    
    outliers = np.random.choice(n_samples, 5, replace=False)
    df.loc[outliers, 'feature_c'] = df['feature_c'].max() * 10
    
    return df

if __name__ == "__main__":
    sample_data = create_sample_data()
    print("Original data shape:", sample_data.shape)
    print("\nMissing values:")
    print(sample_data.isnull().sum())
    
    validation = validate_dataframe(sample_data, 
                                   required_columns=['feature_a', 'feature_b', 'feature_c'],
                                   numeric_columns=['feature_a', 'feature_b', 'feature_c'])
    print("\nValidation result:", validation)
    
    cleaned_data = remove_outliers_iqr(sample_data, 'feature_c')
    print("\nAfter outlier removal shape:", cleaned_data.shape)
    
    imputed_data = handle_missing_values(cleaned_data, strategy='mean')
    print("\nAfter imputation missing values:")
    print(imputed_data.isnull().sum())
    
    imputed_data['feature_a_normalized'] = normalize_minmax(imputed_data, 'feature_a')
    imputed_data['feature_b_standardized'] = standardize_zscore(imputed_data, 'feature_b')
    
    print("\nFirst 5 rows of processed data:")
    print(imputed_data.head())