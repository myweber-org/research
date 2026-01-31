
import os
import time
from pathlib import Path

def clean_old_files(directory, days=7):
    """
    Remove files in the specified directory that are older than the given number of days.
    """
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    cutoff_time = time.time() - (days * 24 * 60 * 60)
    deleted_count = 0
    total_size = 0

    for item in Path(directory).iterdir():
        if item.is_file():
            file_stat = item.stat()
            if file_stat.st_mtime < cutoff_time:
                try:
                    total_size += file_stat.st_size
                    item.unlink()
                    deleted_count += 1
                    print(f"Deleted: {item.name}")
                except OSError as e:
                    print(f"Error deleting {item.name}: {e}")

    print(f"Deleted {deleted_count} files, freed {total_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    target_dir = "/tmp/my_app_cache"
    clean_old_files(target_dir, days=7)