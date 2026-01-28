
import os
import sys

def rename_files_sequentially(directory, prefix="file", extension=".txt"):
    """
    Rename all files in the specified directory with sequential numbering.
    
    Args:
        directory (str): Path to the directory containing files to rename
        prefix (str): Prefix for renamed files
        extension (str): File extension to filter and apply
    
    Returns:
        int: Number of files successfully renamed
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return 0
    
    try:
        files = [f for f in os.listdir(directory) 
                if os.path.isfile(os.path.join(directory, f)) and f.endswith(extension)]
        files.sort()
        
        renamed_count = 0
        for index, filename in enumerate(files, start=1):
            old_path = os.path.join(directory, filename)
            new_filename = f"{prefix}_{index:03d}{extension}"
            new_path = os.path.join(directory, new_filename)
            
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")
                renamed_count += 1
            except OSError as e:
                print(f"Failed to rename {filename}: {e}")
        
        print(f"\nSuccessfully renamed {renamed_count} files.")
        return renamed_count
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory> [prefix] [extension]")
        print("Example: python file_renamer.py ./documents document .pdf")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    file_prefix = sys.argv[2] if len(sys.argv) > 2 else "file"
    file_extension = sys.argv[3] if len(sys.argv) > 3 else ".txt"
    
    rename_files_sequentially(dir_path, file_prefix, file_extension)