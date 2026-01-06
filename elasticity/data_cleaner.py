
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    
    Args:
        input_list (list): List potentially containing duplicates.
    
    Returns:
        list: List with duplicates removed.
    """
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_data(data):
    """
    Clean data by removing duplicates and None values.
    
    Args:
        data (list): Raw data list.
    
    Returns:
        list: Cleaned data list.
    """
    if not isinstance(data, list):
        raise TypeError("Input must be a list")
    
    # Remove None values first
    filtered = [item for item in data if item is not None]
    
    # Remove duplicates
    cleaned = remove_duplicates(filtered)
    
    return cleaned

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, None, 4, 3, 5, None, 1]
    print(f"Original data: {sample_data}")
    print(f"Cleaned data: {clean_data(sample_data)}")