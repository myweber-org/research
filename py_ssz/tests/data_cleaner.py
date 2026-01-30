def remove_duplicates(input_list):
    """
    Remove duplicate items from a list while preserving the original order.
    Returns a new list with duplicates removed.
    """
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result