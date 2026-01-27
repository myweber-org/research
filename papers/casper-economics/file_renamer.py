
import os
import argparse
from datetime import datetime

def rename_files_by_date(directory, prefix="file_"):
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        
        for filename in files:
            filepath = os.path.join(directory, filename)
            mod_time = os.path.getmtime(filepath)
            date_str = datetime.fromtimestamp(mod_time).strftime("%Y%m%d_%H%M%S")
            
            name, ext = os.path.splitext(filename)
            new_filename = f"{prefix}{date_str}{ext}"
            new_filepath = os.path.join(directory, new_filename)
            
            counter = 1
            while os.path.exists(new_filepath):
                new_filename = f"{prefix}{date_str}_{counter}{ext}"
                new_filepath = os.path.join(directory, new_filename)
                counter += 1
            
            os.rename(filepath, new_filepath)
            print(f"Renamed: {filename} -> {new_filename}")
            
        print(f"Successfully renamed {len(files)} files.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename files based on modification date")
    parser.add_argument("directory", help="Directory containing files to rename")
    parser.add_argument("--prefix", default="file_", help="Prefix for renamed files")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.directory):
        rename_files_by_date(args.directory, args.prefix)
    else:
        print(f"Error: {args.directory} is not a valid directory")