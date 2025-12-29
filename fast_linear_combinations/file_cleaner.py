
import os
import shutil
import tempfile
from pathlib import Path

def clean_temp_files(directory, extensions=None, days_old=7):
    """
    Remove temporary files from a directory based on extension and age.
    
    Args:
        directory: Path to directory to clean
        extensions: List of file extensions to remove (e.g., ['.tmp', '.log'])
        days_old: Remove files older than this many days
    """
    if extensions is None:
        extensions = ['.tmp', '.temp', '.log', '.cache']
    
    target_dir = Path(directory)
    if not target_dir.exists():
        raise FileNotFoundError(f"Directory {directory} does not exist")
    
    current_time = time.time()
    cutoff_time = current_time - (days_old * 24 * 60 * 60)
    
    removed_count = 0
    total_size = 0
    
    for file_path in target_dir.rglob('*'):
        if file_path.is_file():
            file_age = file_path.stat().st_mtime
            
            if file_age < cutoff_time or file_path.suffix.lower() in extensions:
                try:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    removed_count += 1
                    total_size += file_size
                    print(f"Removed: {file_path.name} ({file_size} bytes)")
                except Exception as e:
                    print(f"Failed to remove {file_path.name}: {e}")
    
    print(f"\nCleaning complete:")
    print(f"  Files removed: {removed_count}")
    print(f"  Total space freed: {total_size} bytes")
    
    return removed_count, total_size

def create_test_environment():
    """Create test files for demonstration purposes."""
    test_dir = tempfile.mkdtemp(prefix="clean_test_")
    print(f"Created test directory: {test_dir}")
    
    test_files = [
        "document.tmp",
        "backup.log",
        "data.cache",
        "important.txt",
        "config.temp"
    ]
    
    for filename in test_files:
        file_path = Path(test_dir) / filename
        file_path.write_text("Sample content for testing")
    
    return test_dir

if __name__ == "__main__":
    import time
    
    print("Temporary File Cleaner Utility")
    print("=" * 30)
    
    test_dir = create_test_environment()
    
    try:
        print(f"\nCleaning test directory: {test_dir}")
        removed, freed = clean_temp_files(test_dir, days_old=0)
        
        print(f"\nTest completed successfully")
        print(f"Removed {removed} temporary files")
        print(f"Freed {freed} bytes of space")
        
    finally:
        if Path(test_dir).exists():
            shutil.rmtree(test_dir)
            print(f"\nCleaned up test directory: {test_dir}")