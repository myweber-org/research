
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (list or array-like): The dataset containing the column to clean.
    column (int or str): The index or name of the column to process.
    
    Returns:
    tuple: A tuple containing:
        - cleaned_data (list): Data with outliers removed.
        - outlier_indices (list): Indices of removed outliers.
    """
    if not data:
        return [], []
    
    # Convert data to numpy array for easier calculations
    data_array = np.array(data)
    
    # Extract the column values
    if isinstance(column, int):
        column_values = data_array[:, column].astype(float)
    else:
        # If column is a string, we'd need a more complex implementation
        # For simplicity, assuming integer index for this example
        column_values = data_array[:, int(column)].astype(float)
    
    # Calculate Q1, Q3 and IQR
    Q1 = np.percentile(column_values, 25)
    Q3 = np.percentile(column_values, 75)
    IQR = Q3 - Q1
    
    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify non-outliers
    non_outlier_mask = (column_values >= lower_bound) & (column_values <= upper_bound)
    outlier_indices = np.where(~non_outlier_mask)[0].tolist()
    
    # Filter data to remove outliers
    cleaned_data = [data[i] for i in range(len(data)) if non_outlier_mask[i]]
    
    return cleaned_data, outlier_indices

def calculate_basic_stats(data, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    data (list or array-like): The dataset.
    column (int): The index of the column to analyze.
    
    Returns:
    dict: Dictionary containing mean, median, std, min, and max.
    """
    if not data:
        return {}
    
    data_array = np.array(data)
    column_values = data_array[:, column].astype(float)
    
    stats = {
        'mean': np.mean(column_values),
        'median': np.median(column_values),
        'std': np.std(column_values),
        'min': np.min(column_values),
        'max': np.max(column_values),
        'count': len(column_values)
    }
    
    return stats

# Example usage
if __name__ == "__main__":
    # Sample data
    sample_data = [
        [1, 10.5],
        [2, 12.3],
        [3, 11.8],
        [4, 100.0],  # This is an outlier
        [5, 10.9],
        [6, 11.2],
        [7, 9.8],
        [8, 12.1],
        [9, 200.0],  # This is an outlier
        [10, 11.5]
    ]
    
    print("Original data:")
    for row in sample_data:
        print(row)
    
    # Remove outliers from column 1
    cleaned_data, outliers = remove_outliers_iqr(sample_data, 1)
    
    print(f"\nRemoved {len(outliers)} outliers at indices: {outliers}")
    print("\nCleaned data:")
    for row in cleaned_data:
        print(row)
    
    # Calculate statistics
    stats = calculate_basic_stats(cleaned_data, 1)
    print(f"\nStatistics for cleaned data column 1:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
import pandas as pd
import numpy as np
import sys

def clean_data(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Fill missing numeric values with column mean
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
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
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Data cleaning completed. Cleaned data saved to {output_file}")
        print(f"Original rows: {len(pd.read_csv(input_file))}, Cleaned rows: {len(df)}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("Error: Input file is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_cleaner.py <input_file.csv> <output_file.csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    clean_data(input_file, output_file)