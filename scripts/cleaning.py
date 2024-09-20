import pandas as pd
from scripts.logger_config import *

# Set up the logger
logger = setup_logger()


def remove_outliers(df,exclude_columns=None):
    if exclude_columns is None:
        exclude_columns = []
    logger.info("Removing outliers for numerical columns")
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        if column in exclude_columns:
            logger.info(f"Skipping outlier removal for column: {column}")
            continue
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define the bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Remove outliers
        df[column] = df[column].clip(lower_bound, upper_bound)
        
    return df



def remove_missing_values(df, threshold=0.5):
    logger.info(f"Removing missing values with threshold {threshold}")
    # Calculate the threshold number of non-NA values required
    threshold_count = int(threshold * df.shape[1])
    
    df_cleaned = df.dropna(thresh=threshold_count)
    
    return df_cleaned


def remove_categorical_outliers(df, threshold=0.01):
    logger.info(f"Removing categorical outliers with threshold {threshold}")
    for column in df.select_dtypes(include=['object']).columns:
        counts = df[column].value_counts(normalize=True)
        
        rare_categories = counts[counts < threshold].index
        
        df[column] = df[column].replace(rare_categories, 'Other')
        
    return df


def remove_missing_values_categorical(df, fill_value='Unknown'):
    logger.info(f"Removing missing values in categorical columns with fill value {fill_value}")
    for column in df.select_dtypes(include=['object']).columns:
        df[column].fillna(fill_value, inplace=True)
        
    return df
