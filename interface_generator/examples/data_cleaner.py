
def filter_none_values(input_list):
    """
    Return a new list with all None values removed.
    
    Args:
        input_list (list): The list to filter.
    
    Returns:
        list: A new list without None values.
    """
    return [item for item in input_list if item is not None]