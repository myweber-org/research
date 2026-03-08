
import os
import re
import sys

def rename_files(directory, pattern, replacement):
    """
    Rename files in the specified directory based on a regex pattern.
    
    Args:
        directory (str): Path to the directory containing files to rename.
        pattern (str): Regex pattern to match in filenames.
        replacement (str): Replacement string for matched pattern.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        sys.exit(1)
    
    try:
        regex = re.compile(pattern)
    except re.error as e:
        print(f"Error: Invalid regex pattern '{pattern}': {e}")
        sys.exit(1)
    
    renamed_count = 0
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            new_filename = regex.sub(replacement, filename)
            if new_filename != filename:
                new_filepath = os.path.join(directory, new_filename)
                try:
                    os.rename(filepath, new_filepath)
                    print(f"Renamed: '{filename}' -> '{new_filename}'")
                    renamed_count += 1
                except OSError as e:
                    print(f"Error renaming '{filename}': {e}")
    
    print(f"\nRenaming complete. {renamed_count} file(s) renamed.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python file_renamer.py <directory> <pattern> <replacement>")
        print("Example: python file_renamer.py ./files '\\d+' 'NUM'")
        sys.exit(1)
    
    rename_files(sys.argv[1], sys.argv[2], sys.argv[3])