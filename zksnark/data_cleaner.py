def remove_duplicates(data_list):
    seen = set()
    cleaned_data = []
    for item in data_list:
        if item not in seen:
            seen.add(item)
            cleaned_data.append(item)
    return cleaned_data

def clean_data_with_order(data_list, key=None):
    if key is None:
        key = lambda x: x
    seen = set()
    result = []
    for item in data_list:
        identifier = key(item)
        if identifier not in seen:
            seen.add(identifier)
            result.append(item)
    return result

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 5]
    print(remove_duplicates(sample_data))
    
    sample_complex = [{"id": 1}, {"id": 2}, {"id": 1}, {"id": 3}]
    print(clean_data_with_order(sample_complex, key=lambda x: x["id"]))