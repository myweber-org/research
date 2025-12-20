
import os
import re
import sys
from pathlib import Path

def rename_files(directory, pattern, replacement):
    """
    Rename files in the specified directory matching the regex pattern.
    
    Args:
        directory (str): Path to the directory containing files to rename.
        pattern (str): Regex pattern to match in filenames.
        replacement (str): Replacement string for matched pattern.
    """
    try:
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            print(f"Error: Directory '{directory}' does not exist or is not a directory.")
            return
        
        for file_path in dir_path.iterdir():
            if file_path.is_file():
                old_name = file_path.name
                new_name = re.sub(pattern, replacement, old_name)
                
                if new_name != old_name:
                    new_path = file_path.parent / new_name
                    try:
                        file_path.rename(new_path)
                        print(f"Renamed: '{old_name}' -> '{new_name}'")
                    except OSError as e:
                        print(f"Error renaming '{old_name}': {e}")
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python file_renamer.py <directory> <pattern> <replacement>")
        print("Example: python file_renamer.py ./files '\\d+' 'NUM'")
        sys.exit(1)
    
    rename_files(sys.argv[1], sys.argv[2], sys.argv[3])