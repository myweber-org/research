import os
import time
import sys

def remove_old_files(directory, days_old):
    cutoff_time = time.time() - (days_old * 86400)
    removed_count = 0
    error_count = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                file_mtime = os.path.getmtime(file_path)
                if file_mtime < cutoff_time:
                    os.remove(file_path)
                    removed_count += 1
            except Exception as e:
                print(f"Error removing {file_path}: {e}", file=sys.stderr)
                error_count += 1

    return removed_count, error_count

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python file_cleaner.py <directory> <days_old>")
        sys.exit(1)

    target_dir = sys.argv[1]
    try:
        days = int(sys.argv[2])
    except ValueError:
        print("Error: days_old must be an integer")
        sys.exit(1)

    if not os.path.isdir(target_dir):
        print(f"Error: {target_dir} is not a valid directory")
        sys.exit(1)

    removed, errors = remove_old_files(target_dir, days)
    print(f"Removed {removed} files. Encountered {errors} errors.")