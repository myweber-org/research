
import os
import re
import argparse

def rename_files(directory, pattern, replacement, dry_run=False):
    """
    Rename files in the specified directory based on a regex pattern.
    
    Args:
        directory (str): Path to the directory containing files to rename.
        pattern (str): Regex pattern to match in filenames.
        replacement (str): Replacement string for matched pattern.
        dry_run (bool): If True, only show what would be renamed without making changes.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    compiled_pattern = re.compile(pattern)
    renamed_count = 0
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path):
            new_filename = compiled_pattern.sub(replacement, filename)
            
            if new_filename != filename:
                new_file_path = os.path.join(directory, new_filename)
                
                if dry_run:
                    print(f"Would rename: '{filename}' -> '{new_filename}'")
                else:
                    try:
                        os.rename(file_path, new_file_path)
                        print(f"Renamed: '{filename}' -> '{new_filename}'")
                    except OSError as e:
                        print(f"Error renaming '{filename}': {e}")
                        continue
                
                renamed_count += 1
    
    if dry_run:
        print(f"\nDry run complete. {renamed_count} files would be renamed.")
    else:
        print(f"\nRenaming complete. {renamed_count} files renamed.")

def main():
    parser = argparse.ArgumentParser(description="Rename files in a directory using regex patterns.")
    parser.add_argument("directory", help="Directory containing files to rename")
    parser.add_argument("pattern", help="Regex pattern to match in filenames")
    parser.add_argument("replacement", help="Replacement string for matched pattern")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be renamed without making changes")
    
    args = parser.parse_args()
    
    rename_files(args.directory, args.pattern, args.replacement, args.dry_run)

if __name__ == "__main__":
    main()