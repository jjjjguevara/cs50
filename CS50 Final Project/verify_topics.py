import requests
import json
from pathlib import Path

def check_filesystem():
    """Check DITA files in the filesystem"""
    base_path = Path('app/dita/topics')
    print("\nChecking filesystem:")

    subdirs = ['acoustics', 'articles', 'audio', 'journals', 'abstracts']
    for subdir in subdirs:
        path = base_path / subdir
        if path.exists():
            print(f"\n{subdir} directory:")
            for file in path.glob('*.dita'):
                print(f"  - {file.name}")
        else:
            print(f"{subdir} directory not found")

def check_api():
    """Check topics via API"""
    print("\nChecking API endpoints:")

    # Check topics endpoint
    response = requests.get('http://localhost:5001/api/topics')
    print("\nTopics endpoint response:")
    print(json.dumps(response.json(), indent=2))

    # Check test topics endpoint
    response = requests.get('http://localhost:5001/api/test-topics')
    print("\nTest topics endpoint response:")
    print(json.dumps(response.json(), indent=2))

if __name__ == '__main__':
    print("Verifying DITA topics...")
    check_filesystem()
    check_api()
