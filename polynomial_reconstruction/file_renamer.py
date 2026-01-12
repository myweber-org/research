
import os
import time
from pathlib import Path

def add_timestamp_to_filename(filepath):
    """
    Rename a file by adding a timestamp prefix to its filename.
    Keeps the original extension.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    new_filename = f"{timestamp}_{path.name}"
    new_path = path.parent / new_filename
    
    try:
        path.rename(new_path)
        return str(new_path)
    except Exception as e:
        raise RuntimeError(f"Failed to rename file: {e}")

def batch_rename_files(directory, extension=None):
    """
    Add timestamp prefix to all files in a directory.
    Optional filter by file extension.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Invalid directory: {directory}")
    
    renamed_files = []
    for item in dir_path.iterdir():
        if item.is_file():
            if extension is None or item.suffix.lower() == extension.lower():
                try:
                    new_path = add_timestamp_to_filename(str(item))
                    renamed_files.append(new_path)
                except Exception as e:
                    print(f"Error renaming {item.name}: {e}")
    
    return renamed_files

if __name__ == "__main__":
    # Example usage
    test_file = "example_document.txt"
    
    # Create a test file if it doesn't exist
    if not Path(test_file).exists():
        with open(test_file, 'w') as f:
            f.write("Test content")
    
    try:
        result = add_timestamp_to_filename(test_file)
        print(f"Renamed to: {result}")
    except Exception as e:
        print(f"Error: {e}")