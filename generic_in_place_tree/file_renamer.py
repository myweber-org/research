
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