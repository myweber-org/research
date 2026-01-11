
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    Returns a new list with unique elements.
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
    If key_func is None, uses the element itself.
    """
    seen = set()
    cleaned = []
    for item in data:
        key = key_func(item) if key_func else item
        if key not in seen:
            seen.add(key)
            cleaned.append(item)
    return cleaned

if __name__ == "__main__":
    sample = [1, 2, 2, 3, 4, 4, 5]
    print("Original:", sample)
    print("Cleaned:", remove_duplicates(sample))
    
    sample_dicts = [{"id": 1}, {"id": 2}, {"id": 1}, {"id": 3}]
    print("Original dicts:", sample_dicts)
    print("Cleaned by id:", clean_data_with_key(sample_dicts, key_func=lambda x: x["id"]))