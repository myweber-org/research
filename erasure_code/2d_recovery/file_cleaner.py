
import os
import time
from pathlib import Path

def cleanup_old_files(directory_path, days_old):
    cutoff_time = time.time() - (days_old * 86400)
    target_path = Path(directory_path)

    if not target_path.exists() or not target_path.is_dir():
        print(f"Directory {directory_path} does not exist or is not a directory.")
        return

    for item in target_path.rglob('*'):
        if item.is_file():
            if item.stat().st_mtime < cutoff_time:
                try:
                    item.unlink()
                    print(f"Deleted: {item}")
                except OSError as e:
                    print(f"Error deleting {item}: {e}")

if __name__ == "__main__":
    temp_dir = "/tmp/test_cleanup"
    cleanup_old_files(temp_dir, 7)