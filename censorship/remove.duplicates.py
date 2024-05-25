import os
import hashlib

def get_file_hash(file_path):
    """Generate MD5 hash for a given file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def find_duplicates(directory):
    """Find and remove duplicate files in a directory and its subdirectories."""
    files_seen = {}
    duplicates = []

    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            file_size = os.path.getsize(file_path)
            file_id = (filename, file_size)
            
            if file_id not in files_seen:
                files_seen[file_id] = file_path
            else:
                # Verify by hashing if sizes match
                if get_file_hash(file_path) == get_file_hash(files_seen[file_id]):
                    duplicates.append(file_path)

    for duplicate in duplicates:
        print(f"Removing duplicate file: {duplicate}")
        os.remove(duplicate)

if __name__ == "__main__":
    directory = input("Enter the directory path: ")
    find_duplicates(directory)
    print("Duplicate files have been removed.")