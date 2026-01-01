import os
import shutil
import tempfile
from pathlib import Path

def clean_temp_files(directory: str, days_old: int = 7, dry_run: bool = False) -> list:
    """
    Remove temporary files older than a specified number of days.
    Returns list of removed files.
    """
    import time
    from datetime import datetime, timedelta

    cutoff_time = time.time() - (days_old * 86400)
    removed_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = Path(root) / file
            try:
                if file_path.stat().st_mtime < cutoff_time:
                    if dry_run:
                        print(f"[DRY RUN] Would remove: {file_path}")
                    else:
                        if file_path.is_file():
                            file_path.unlink()
                            removed_files.append(str(file_path))
            except (OSError, PermissionError) as e:
                print(f"Error processing {file_path}: {e}")

    return removed_files

def create_sample_temp_files() -> str:
    """Create sample temporary files for testing."""
    temp_dir = tempfile.mkdtemp(prefix="test_clean_")
    
    for i in range(5):
        temp_file = Path(temp_dir) / f"temp_file_{i}.tmp"
        temp_file.write_text(f"Temporary content {i}")
    
    return temp_dir

if __name__ == "__main__":
    test_dir = create_sample_temp_files()
    print(f"Created test directory: {test_dir}")
    
    result = clean_temp_files(test_dir, days_old=0, dry_run=True)
    print(f"Files to be removed: {len(result)}")
    
    shutil.rmtree(test_dir)
    print("Test cleanup completed.")