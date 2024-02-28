def linear_norm_global_percentile(image, min_val=-14.11, max_val=-3.82):
    """
    Normalize the image linearly using global 2nd and 98th percentiles as 
    min and max values.
    """
    return (image - min_val) / (max_val - min_val)

def linear_norm_global_minmax(image, min_val=-19.63, max_val=1.43):
    """
    Normalize the image linearly using global min and max values.
    """
    return (image - min_val) / (max_val - min_val)

def global_standardization(image, global_mean=-8.51, global_std=2.46):
    """
    Normalize the image using global mean and standard deviation in order to have
    a mean of 0 and a standard deviation of 1.
    """
    return (image - global_mean) / global_std

