import numpy as np

def linear_norm_global_percentile(image, min_val=0, max_val=3481.5):
    """
    Normalize the image linearly using global 2nd and 98th percentiles as 
    min and max values.
    """
    return (image - min_val) / (max_val - min_val)

def linear_norm_global_minmax(image, min_val=0, max_val=6098.5):
    """
    Normalize the image linearly using global min and max values.
    """
    return (image - min_val) / (max_val - min_val)

def global_standardization(image, global_mean=573.88, global_std=981.79):
    """
    Normalize the image using global mean and standard deviation in order to have
    a mean of 0 and a standard deviation of 1.
    """
    return (image - global_mean) / global_std

