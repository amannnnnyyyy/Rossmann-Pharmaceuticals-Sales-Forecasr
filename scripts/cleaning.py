import pandas as pd
from scripts.logger_config import *

# Set up the logger
logger = setup_logger()

def remove_outliers(df):
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define the bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Remove outliers
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        
    return df



def remove_missing_values(df, threshold=0.5):
    # Calculate the threshold number of non-NA values required
    threshold_count = int(threshold * df.shape[1])
    
    # Remove rows that do not meet the threshold
    df_cleaned = df.dropna(thresh=threshold_count)
    
    return df_cleaned


def remove_categorical_outliers(df, threshold=0.01):
    for column in df.select_dtypes(include=['object']).columns:
        # Calculate the frequency of each category
        counts = df[column].value_counts(normalize=True)
        
        # Identify categories that are less frequent than the threshold
        rare_categories = counts[counts < threshold].index
        
        # Remove rare categories
        df[column] = df[column].replace(rare_categories, 'Other')
        
    return df


def remove_missing_values_categorical(df, fill_value='Unknown'):
    for column in df.select_dtypes(include=['object']).columns:
        df[column].fillna(fill_value, inplace=True)
        
    return df
