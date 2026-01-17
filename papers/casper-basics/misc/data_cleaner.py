import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = df.select_dtypes(exclude=[np.number]).columns
        
    def detect_outliers_iqr(self, column, threshold=1.5):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        return outliers.index.tolist()
    
    def remove_outliers(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.numeric_columns
            
        outlier_indices = []
        for col in columns:
            if col in self.numeric_columns:
                outlier_indices.extend(self.detect_outliers_iqr(col, threshold))
        
        unique_outliers = list(set(outlier_indices))
        cleaned_df = self.df.drop(index=unique_outliers)
        return cleaned_df, len(unique_outliers)
    
    def impute_missing_median(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
            
        imputed_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns and imputed_df[col].isnull().any():
                median_value = imputed_df[col].median()
                imputed_df[col].fillna(median_value, inplace=True)
        return imputed_df
    
    def impute_missing_mode(self, columns=None):
        if columns is None:
            columns = self.categorical_columns
            
        imputed_df = self.df.copy()
        for col in columns:
            if col in self.categorical_columns and imputed_df[col].isnull().any():
                mode_value = imputed_df[col].mode()[0]
                imputed_df[col].fillna(mode_value, inplace=True)
        return imputed_df
    
    def standardize_numeric(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
            
        standardized_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                mean = standardized_df[col].mean()
                std = standardized_df[col].std()
                if std > 0:
                    standardized_df[col] = (standardized_df[col] - mean) / std
        return standardized_df
    
    def get_summary(self):
        summary = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'numeric_columns': list(self.numeric_columns),
            'categorical_columns': list(self.categorical_columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary