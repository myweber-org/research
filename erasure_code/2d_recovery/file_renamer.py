
import os
import sys
from datetime import datetime

def rename_files_with_timestamp(directory_path):
    """
    Rename all files in the specified directory by adding a timestamp prefix.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        return False
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    renamed_count = 0
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        if os.path.isfile(file_path):
            name, ext = os.path.splitext(filename)
            new_filename = f"{timestamp}_{name}{ext}"
            new_file_path = os.path.join(directory_path, new_filename)
            
            try:
                os.rename(file_path, new_file_path)
                print(f"Renamed: {filename} -> {new_filename}")
                renamed_count += 1
            except OSError as e:
                print(f"Failed to rename {filename}: {e}")
    
    print(f"Total files renamed: {renamed_count}")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python file_renamer.py <directory_path>")
        sys.exit(1)
    
    target_directory = sys.argv[1]
    rename_files_with_timestamp(target_directory)
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
    try:
        if not os.path.isdir(directory):
            print(f"Error: Directory '{directory}' does not exist.")
            return False

        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        renamed_count = 0

        for filename in files:
            new_name = re.sub(pattern, replacement, filename)
            
            if new_name != filename:
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_name)
                
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: '{filename}' -> '{new_name}'")
                    renamed_count += 1
                except OSError as e:
                    print(f"Error renaming '{filename}': {e}")

        print(f"Renaming complete. {renamed_count} files renamed.")
        return True

    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def main():
    if len(sys.argv) != 4:
        print("Usage: python file_renamer.py <directory> <pattern> <replacement>")
        print("Example: python file_renamer.py ./files '\\d+' 'NUM'")
        sys.exit(1)

    directory = sys.argv[1]
    pattern = sys.argv[2]
    replacement = sys.argv[3]

    success = rename_files(directory, pattern, replacement)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()