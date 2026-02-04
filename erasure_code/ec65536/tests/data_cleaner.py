
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    
    Args:
        input_list: A list of elements (must be hashable)
    
    Returns:
        A new list with duplicates removed
    """
    seen = set()
    result = []
    
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    
    return result

def clean_data_with_key(data, key_func=None):
    """
    Remove duplicates based on a key function.
    
    Args:
        data: List of items to deduplicate
        key_func: Function to extract comparison key (default: identity)
    
    Returns:
        Deduplicated list preserving order
    """
    if key_func is None:
        return remove_duplicates(data)
    
    seen = set()
    result = []
    
    for item in data:
        key = key_func(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    
    return result

if __name__ == "__main__":
    # Example usage
    sample_data = [1, 2, 2, 3, 4, 4, 5, 1]
    cleaned = remove_duplicates(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")
    
    # Example with custom key
    data_dicts = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
        {"id": 1, "name": "Alice"},
        {"id": 3, "name": "Charlie"}
    ]
    
    cleaned_dicts = clean_data_with_key(data_dicts, key_func=lambda x: x["id"])
    print(f"\nOriginal dicts: {data_dicts}")
    print(f"Cleaned dicts: {cleaned_dicts}")