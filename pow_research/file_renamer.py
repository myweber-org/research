
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
import os
import sys

def rename_files_with_sequence(directory, prefix="file", extension=".txt"):
    """
    Rename all files in the given directory with sequential numbering.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return False
    
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files.sort()
    
    for index, filename in enumerate(files, start=1):
        old_path = os.path.join(directory, filename)
        new_filename = f"{prefix}_{index:03d}{extension}"
        new_path = os.path.join(directory, new_filename)
        
        try:
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")
        except OSError as e:
            print(f"Failed to rename {filename}: {e}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory> [prefix] [extension]")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    prefix_arg = sys.argv[2] if len(sys.argv) > 2 else "file"
    extension_arg = sys.argv[3] if len(sys.argv) > 3 else ".txt"
    
    rename_files_with_sequence(target_dir, prefix_arg, extension_arg)