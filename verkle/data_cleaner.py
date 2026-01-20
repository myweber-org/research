
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataset(input_path, output_path):
    try:
        df = pd.read_csv(input_path)
        original_shape = df.shape
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df = remove_outliers_iqr(df, col)
        
        df = df.dropna()
        final_shape = df.shape
        
        df.to_csv(output_path, index=False)
        
        print(f"Original dataset shape: {original_shape}")
        print(f"Cleaned dataset shape: {final_shape}")
        print(f"Removed {original_shape[0] - final_shape[0]} rows")
        print(f"Cleaned data saved to: {output_path}")
        
        return df
    except FileNotFoundError:
        print(f"Error: File '{input_path}' not found")
        return None
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return None

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    cleaned_df = clean_dataset(input_file, output_file)