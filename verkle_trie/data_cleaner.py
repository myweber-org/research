def remove_duplicates(input_list):
    """
    Remove duplicate items from a list while preserving the original order.
    """
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_numeric_strings(string_list):
    """
    Clean a list of strings by converting numeric strings to integers.
    Non-numeric strings are kept as-is.
    """
    cleaned = []
    for s in string_list:
        s_stripped = s.strip()
        if s_stripped.isdigit():
            cleaned.append(int(s_stripped))
        else:
            cleaned.append(s)
    return cleaned

def filter_by_type(data_list, data_type):
    """
    Filter a list to include only items of a specific type.
    """
    return [item for item in data_list if isinstance(item, data_type)]

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, "4", "4", "five", 5.0, 5.0]
    print("Original:", sample_data)
    
    unique_data = remove_duplicates(sample_data)
    print("After removing duplicates:", unique_data)
    
    cleaned_data = clean_numeric_strings(unique_data)
    print("After cleaning numeric strings:", cleaned_data)
    
    integers_only = filter_by_type(cleaned_data, int)
    print("Integers only:", integers_only)