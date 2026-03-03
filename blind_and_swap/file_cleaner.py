
import os
import re
import sys

def normalize_filename(filename):
    """Convert filename to lowercase, replace spaces with underscores, remove special chars."""
    name, ext = os.path.splitext(filename)
    name = name.lower()
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'[-\s]+', '_', name)
    return name + ext.lower()

def clean_directory(directory_path):
    """Rename all files in directory with normalized names."""
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        return False

    for filename in os.listdir(directory_path):
        old_path = os.path.join(directory_path, filename)
        if os.path.isfile(old_path):
            new_name = normalize_filename(filename)
            new_path = os.path.join(directory_path, new_name)
            if old_path != new_path:
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {filename} -> {new_name}")
                except OSError as e:
                    print(f"Failed to rename {filename}: {e}")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python file_cleaner.py <directory_path>")
        sys.exit(1)
    target_dir = sys.argv[1]
    clean_directory(target_dir)