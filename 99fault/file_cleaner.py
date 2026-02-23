
import os
import glob
import sys

def clean_temp_files(directory, patterns):
    """
    Remove temporary files matching given patterns in the specified directory.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return False

    removed_files = []
    for pattern in patterns:
        search_path = os.path.join(directory, pattern)
        for file_path in glob.glob(search_path):
            try:
                os.remove(file_path)
                removed_files.append(file_path)
                print(f"Removed: {file_path}")
            except OSError as e:
                print(f"Error removing {file_path}: {e}")

    if removed_files:
        print(f"Cleaned {len(removed_files)} temporary file(s).")
    else:
        print("No temporary files found to clean.")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_cleaner.py <directory>")
        sys.exit(1)

    target_dir = sys.argv[1]
    temp_patterns = ['*.tmp', '*.temp', '~*', '*.bak']
    clean_temp_files(target_dir, temp_patterns)