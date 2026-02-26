
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    
    Args:
        input_list: A list containing elements (must be hashable).
    
    Returns:
        A new list with duplicates removed.
    """
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_data_with_threshold(data, threshold=None):
    """
    Clean data by removing duplicates and optionally filtering by threshold.
    
    Args:
        data: List of numerical values.
        threshold: Optional minimum value to keep.
    
    Returns:
        Cleaned list.
    """
    unique_data = remove_duplicates(data)
    if threshold is not None:
        unique_data = [x for x in unique_data if x >= threshold]
    return sorted(unique_data)

if __name__ == "__main__":
    sample_data = [3, 1, 2, 3, 4, 2, 5, 1, 6]
    print("Original:", sample_data)
    cleaned = remove_duplicates(sample_data)
    print("Cleaned:", cleaned)
    
    numeric_data = [10, 5, 15, 5, 20, 10, 15]
    filtered = clean_data_with_threshold(numeric_data, threshold=10)
    print("Filtered (>=10):", filtered)