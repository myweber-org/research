
import os
import re
import sys

def rename_files(directory, pattern, replacement):
    """
    Rename files in the specified directory based on a regex pattern.
    
    Args:
        directory (str): Path to the directory containing files to rename.
        pattern (str): Regex pattern to match in filenames.
        replacement (str): String to replace matched pattern with.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        sys.exit(1)
    
    try:
        regex = re.compile(pattern)
    except re.error as e:
        print(f"Error: Invalid regex pattern - {e}")
        sys.exit(1)
    
    renamed_count = 0
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path):
            new_filename = regex.sub(replacement, filename)
            
            if new_filename != filename:
                new_file_path = os.path.join(directory, new_filename)
                
                if os.path.exists(new_file_path):
                    print(f"Warning: Skipping '{filename}' -> '{new_filename}' (target already exists)")
                    continue
                
                try:
                    os.rename(file_path, new_file_path)
                    print(f"Renamed: '{filename}' -> '{new_filename}'")
                    renamed_count += 1
                except OSError as e:
                    print(f"Error renaming '{filename}': {e}")
    
    print(f"\nRenaming complete. {renamed_count} files renamed.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python file_renamer.py <directory> <pattern> <replacement>")
        print("Example: python file_renamer.py ./files '\\d+' 'NUM'")
        sys.exit(1)
    
    rename_files(sys.argv[1], sys.argv[2], sys.argv[3])