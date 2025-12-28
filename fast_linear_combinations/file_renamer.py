
import os
import re
import argparse

def rename_files(directory, pattern, replacement, dry_run=False):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    compiled_pattern = re.compile(pattern)

    for filename in files:
        new_name = compiled_pattern.sub(replacement, filename)
        if new_name != filename:
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            if dry_run:
                print(f"Would rename '{filename}' to '{new_name}'")
            else:
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed '{filename}' to '{new_name}'")
                except OSError as e:
                    print(f"Error renaming '{filename}': {e}")

def main():
    parser = argparse.ArgumentParser(description="Rename files in a directory using regex pattern matching.")
    parser.add_argument("directory", help="Directory containing files to rename")
    parser.add_argument("pattern", help="Regex pattern to match in filenames")
    parser.add_argument("replacement", help="Replacement string for matched pattern")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be renamed without making changes")

    args = parser.parse_args()
    rename_files(args.directory, args.pattern, args.replacement, args.dry_run)

if __name__ == "__main__":
    main()