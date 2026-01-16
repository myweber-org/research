
def remove_duplicates(input_list):
    """
    Remove duplicate items from a list while preserving order.
    Returns a new list with unique elements.
    """
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_numeric_data(values, remove_none=True):
    """
    Clean a list of numeric values by removing None values
    and converting strings to floats/integers.
    """
    cleaned = []
    for val in values:
        if val is None and remove_none:
            continue
        if isinstance(val, str):
            try:
                if '.' in val:
                    val = float(val)
                else:
                    val = int(val)
            except ValueError:
                continue
        cleaned.append(val)
    return cleaned

def filter_by_threshold(data, threshold, key=None):
    """
    Filter data based on a threshold value.
    If key is provided, it should be a function to extract comparison value.
    """
    if key is None:
        key = lambda x: x
    
    return [item for item in data if key(item) >= threshold]