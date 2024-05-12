def linear_norm_global_percentile(image, min_val=-12.381244628070895, max_val=-3.7340425864436555):
    """
    Normalize the image linearly using global 2nd and 98th percentiles as 
    min and max values.
    """
    return (image - min_val) / (max_val - min_val)

def linear_norm_global_minmax(image, min_val=-18.34905638353096, max_val=4.438379134930133):
    """
    Normalize the image linearly using global min and max values.
    """
    return (image - min_val) / (max_val - min_val)

def global_standardization(image, global_mean=-7.494522479755267, global_std=2.0433425028393675):
    """
    Normalize the image using global mean and standard deviation in order to have
    a mean of 0 and a standard deviation of 1.
    """
    return (image - global_mean) / global_std

