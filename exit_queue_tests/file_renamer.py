
import os
import datetime
import argparse

def rename_files_by_date(directory, prefix="", dry_run=False):
    """
    Rename files in a directory to include their creation date.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        if os.path.isfile(filepath):
            try:
                creation_time = os.path.getctime(filepath)
                creation_date = datetime.datetime.fromtimestamp(creation_time).strftime('%Y%m%d')
                
                name, ext = os.path.splitext(filename)
                new_filename = f"{prefix}{creation_date}_{name}{ext}"
                new_filepath = os.path.join(directory, new_filename)
                
                if dry_run:
                    print(f"Would rename: {filename} -> {new_filename}")
                else:
                    os.rename(filepath, new_filepath)
                    print(f"Renamed: {filename} -> {new_filename}")
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename files with creation date prefix.")
    parser.add_argument("directory", help="Directory containing files to rename")
    parser.add_argument("--prefix", default="", help="Optional prefix for renamed files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be renamed without making changes")
    
    args = parser.parse_args()
    rename_files_by_date(args.directory, args.prefix, args.dry_run)