
import os
import re
import argparse

def rename_files(directory, pattern, replacement):
    try:
        files = os.listdir(directory)
        renamed_count = 0
        
        for filename in files:
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                new_name = re.sub(pattern, replacement, filename)
                if new_name != filename:
                    new_path = os.path.join(directory, new_name)
                    os.rename(file_path, new_path)
                    print(f"Renamed: {filename} -> {new_name}")
                    renamed_count += 1
        
        print(f"Total files renamed: {renamed_count}")
        
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename files in a directory using regex pattern")
    parser.add_argument("directory", help="Directory containing files to rename")
    parser.add_argument("pattern", help="Regex pattern to match in filenames")
    parser.add_argument("replacement", help="Replacement string for matched pattern")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
    else:
        rename_files(args.directory, args.pattern, args.replacement)