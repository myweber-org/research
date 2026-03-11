
import os
import sys

def batch_rename_files(directory, prefix, start_number=1, extension=None):
    """
    Rename all files in a directory with sequential numbering.
    
    Args:
        directory (str): Path to directory containing files
        prefix (str): Prefix for renamed files
        start_number (int): Starting number for sequence
        extension (str): Filter by file extension (e.g., 'txt', 'jpg')
    """
    try:
        if not os.path.exists(directory):
            print(f"Error: Directory '{directory}' does not exist.")
            return
        
        files = os.listdir(directory)
        
        if extension:
            files = [f for f in files if f.lower().endswith(f'.{extension.lower()}')]
        
        files.sort()
        
        counter = start_number
        
        for filename in files:
            old_path = os.path.join(directory, filename)
            
            if os.path.isfile(old_path):
                file_ext = os.path.splitext(filename)[1]
                new_filename = f"{prefix}_{counter:03d}{file_ext}"
                new_path = os.path.join(directory, new_filename)
                
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {filename} -> {new_filename}")
                    counter += 1
                except Exception as e:
                    print(f"Failed to rename {filename}: {e}")
        
        print(f"\nRenaming complete. {counter - start_number} files renamed.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python file_renamer.py <directory> <prefix> [start_number] [extension]")
        print("Example: python file_renamer.py ./photos vacation_ 1 jpg")
        sys.exit(1)
    
    directory = sys.argv[1]
    prefix = sys.argv[2]
    start_number = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    extension = sys.argv[4] if len(sys.argv) > 4 else None
    
    batch_rename_files(directory, prefix, start_number, extension)
import os
import datetime

def rename_files_with_timestamp(directory_path, file_extension=None):
    """
    Rename files in a directory by adding a timestamp prefix.
    
    Args:
        directory_path (str): Path to the directory containing files to rename.
        file_extension (str, optional): Specific file extension to filter files.
                                        If None, all files will be processed.
    """
    try:
        if not os.path.isdir(directory_path):
            print(f"Error: Directory '{directory_path}' does not exist.")
            return
        
        files = os.listdir(directory_path)
        
        for filename in files:
            file_path = os.path.join(directory_path, filename)
            
            if os.path.isfile(file_path):
                if file_extension and not filename.endswith(file_extension):
                    continue
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                name_part, extension = os.path.splitext(filename)
                new_filename = f"{timestamp}_{name_part}{extension}"
                new_file_path = os.path.join(directory_path, new_filename)
                
                os.rename(file_path, new_file_path)
                print(f"Renamed: {filename} -> {new_filename}")
    
    except PermissionError:
        print("Error: Permission denied. Check file access permissions.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    target_directory = "/path/to/your/files"
    rename_files_with_timestamp(target_directory, ".txt")