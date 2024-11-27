import os
from pathlib import Path

def check_paths():
    # Get the current directory
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")

    # Expected paths
    app_dir = current_dir / 'app'
    topics_dir = app_dir / 'dita' / 'topics'

    # Check all potential locations
    for subdir in ['abstracts', 'acoustics', 'articles', 'audio', 'journals']:
        check_dir = topics_dir / subdir
        print(f"\nChecking directory: {check_dir}")
        print(f"Directory exists: {check_dir.exists()}")

        if check_dir.exists():
            print("Files in directory:")
            for file in check_dir.glob('*.dita'):
                print(f"  - {file.name}")

if __name__ == '__main__':
    check_paths()
