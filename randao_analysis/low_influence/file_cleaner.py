
import os
import glob
import sys

def clean_temp_files(directory, patterns):
    """
    Remove temporary files matching given patterns from a directory.
    
    Args:
        directory: Path to the directory to clean.
        patterns: List of file patterns to match (e.g., ['*.tmp', 'temp_*']).
    
    Returns:
        Number of files removed.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return 0
    
    removed_count = 0
    for pattern in patterns:
        search_path = os.path.join(directory, pattern)
        for file_path in glob.glob(search_path):
            try:
                os.remove(file_path)
                print(f"Removed: {file_path}")
                removed_count += 1
            except OSError as e:
                print(f"Error removing {file_path}: {e}")
    
    return removed_count

def main():
    if len(sys.argv) < 3:
        print("Usage: python file_cleaner.py <directory> <pattern1> [pattern2 ...]")
        sys.exit(1)
    
    directory = sys.argv[1]
    patterns = sys.argv[2:]
    
    count = clean_temp_files(directory, patterns)
    print(f"Total files removed: {count}")

if __name__ == "__main__":
    main()