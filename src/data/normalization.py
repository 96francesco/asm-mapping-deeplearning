def planet_norm(image, min_val=206.5, max_val=3770.5):
    return (image - min_val) / (max_val - min_val)

def s1_norm(image, min_val=-12.381244628070895, max_val=-3.7340425864436555):
    return (image - min_val) / (max_val - min_val)