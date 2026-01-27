
import os
import sys

def rename_files_with_sequence(directory, prefix="file", start_number=1):
    """
    Rename all files in the specified directory with sequential numbering.
    
    Args:
        directory (str): Path to the directory containing files to rename
        prefix (str): Prefix for renamed files (default: "file")
        start_number (int): Starting number for sequence (default: 1)
    """
    try:
        if not os.path.isdir(directory):
            print(f"Error: Directory '{directory}' does not exist.")
            return False
        
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        files.sort()
        
        if not files:
            print("No files found in the directory.")
            return True
        
        print(f"Found {len(files)} files to rename.")
        
        for index, filename in enumerate(files, start=start_number):
            file_extension = os.path.splitext(filename)[1]
            new_name = f"{prefix}_{index:03d}{file_extension}"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_name}")
            except OSError as e:
                print(f"Error renaming {filename}: {e}")
        
        print("File renaming completed.")
        return True
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory> [prefix] [start_number]")
        print("Example: python file_renamer.py ./photos vacation 1")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else "file"
    start_num = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    success = rename_files_with_sequence(dir_path, prefix, start_num)
    sys.exit(0 if success else 1)