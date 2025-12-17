
import pandas as pd
import numpy as np

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

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_column(cleaned_df, col)
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'feature1': np.random.normal(100, 15, 200),
        'feature2': np.random.exponential(50, 200),
        'category': np.random.choice(['A', 'B', 'C'], 200)
    }
    df = pd.DataFrame(sample_data)
    numeric_cols = ['feature1', 'feature2']
    result = clean_dataset(df, numeric_cols)
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {result.shape}")
    print(result.head())
import pandas as pd
import numpy as np
from datetime import datetime

def clean_data(input_file, output_file):
    """
    Clean and preprocess data from a CSV file.
    """
    try:
        df = pd.read_csv(input_file)
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Fill missing numeric values with column mean
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.mean()))
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Remove outliers using IQR method for numeric columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Add processing timestamp
        df['processed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Data cleaning completed. Cleaned data saved to {output_file}")
        print(f"Original rows: {len(pd.read_csv(input_file))}, Cleaned rows: {len(df)}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    cleaned_df = clean_data(input_csv, output_csv)
    
    if cleaned_df is not None:
        print("Data cleaning summary:")
        print(f"Columns: {list(cleaned_df.columns)}")
        print(f"Data types:\n{cleaned_df.dtypes}")