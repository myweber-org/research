
import os
import sys
import argparse

def rename_files_with_sequence(directory, prefix, start_number=1):
    """
    Rename all files in a directory with sequential numbering.
    Keeps original file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return False
    
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files.sort()
    
    renamed_count = 0
    for index, filename in enumerate(files, start=start_number):
        file_extension = os.path.splitext(filename)[1]
        new_filename = f"{prefix}_{index:03d}{file_extension}"
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        
        try:
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")
            renamed_count += 1
        except OSError as e:
            print(f"Failed to rename {filename}: {e}")
    
    print(f"\nTotal files renamed: {renamed_count}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Rename files with sequential numbering.')
    parser.add_argument('directory', help='Directory containing files to rename')
    parser.add_argument('prefix', help='Prefix for renamed files')
    parser.add_argument('--start', type=int, default=1, help='Starting number (default: 1)')
    
    args = parser.parse_args()
    
    success = rename_files_with_sequence(args.directory, args.prefix, args.start)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()