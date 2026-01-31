
import os
import shutil
import tempfile
from pathlib import Path

def clean_temporary_files(directory_path, extensions=None, days_old=7):
    """
    Remove temporary files from a specified directory.
    Files can be filtered by extension and age.
    """
    if extensions is None:
        extensions = ['.tmp', '.temp', '.log', '.cache']
    
    target_dir = Path(directory_path)
    if not target_dir.exists() or not target_dir.is_dir():
        raise ValueError(f"Invalid directory: {directory_path}")
    
    current_time = os.path.getctime(target_dir)
    removed_count = 0
    total_size = 0
    
    for item in target_dir.rglob('*'):
        if item.is_file():
            file_age = current_time - os.path.getctime(item)
            if file_age > days_old * 86400:
                if any(item.suffix.lower() == ext.lower() for ext in extensions):
                    try:
                        file_size = item.stat().st_size
                        item.unlink()
                        removed_count += 1
                        total_size += file_size
                        print(f"Removed: {item.name}")
                    except OSError as e:
                        print(f"Failed to remove {item.name}: {e}")
    
    print(f"Cleaning complete. Removed {removed_count} files, freed {total_size} bytes.")
    return removed_count, total_size

def create_sample_temporary_files(test_dir):
    """Create sample temporary files for testing."""
    test_dir = Path(test_dir)
    test_dir.mkdir(exist_ok=True)
    
    sample_files = [
        "document.tmp",
        "backup.temp",
        "error.log",
        "data.cache",
        "important.txt"
    ]
    
    for filename in sample_files:
        file_path = test_dir / filename
        file_path.touch()
    
    return test_dir

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Testing with directory: {tmpdir}")
        test_directory = create_sample_temporary_files(tmpdir)
        result = clean_temporary_files(test_directory)
        print(f"Result: {result}")