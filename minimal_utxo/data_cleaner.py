
def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    
    Args:
        data_list (list): Input list potentially containing duplicates.
    
    Returns:
        list: List with duplicates removed.
    """
    seen = set()
    result = []
    for item in data_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_numeric_data(values, threshold=None):
    """
    Clean numeric data by removing None values and optionally filtering by threshold.
    
    Args:
        values (list): List of numeric values.
        threshold (float, optional): Maximum allowed value. Values above threshold are removed.
    
    Returns:
        list: Cleaned numeric list.
    """
    cleaned = []
    for val in values:
        if val is None:
            continue
        try:
            num = float(val)
            if threshold is not None and num > threshold:
                continue
            cleaned.append(num)
        except (ValueError, TypeError):
            continue
    return cleaned

if __name__ == "__main__":
    # Example usage
    sample_data = [1, 2, 2, 3, 4, 4, 5]
    cleaned = remove_duplicates(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")
    
    numeric_data = [10, 20, None, 30, "40", 50, 1000]
    threshold_cleaned = clean_numeric_data(numeric_data, threshold=100)
    print(f"Numeric cleaned: {threshold_cleaned}")