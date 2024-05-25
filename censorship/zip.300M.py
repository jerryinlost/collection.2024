import os
import rarfile
import subprocess
import sys

# Set the password for the RAR files
RAR_PASSWORD = "your_password"

# Set the threshold size in bytes (300 MB)
THRESHOLD_SIZE = 300 * 1024 * 1024

def get_size(start_path):
    """Calculate the total size of the files in a directory."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

def rar_files(directory, files, rar_file_path):
    """Create a RAR archive with the given files."""
    with rarfile.RarFile(rar_file_path, mode='w', pwd=RAR_PASSWORD) as rf:
        for file in files:
            rf.write(file, arcname=os.path.basename(file))

def main(directory):
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return

    # List all items in the directory and sort them alphabetically
    items = sorted(os.listdir(directory))
    
    current_batch = []
    current_batch_size = 0
    batch_index = 1

    for item in items:
        item_path = os.path.join(directory, item)
        
        if os.path.isfile(item_path):
            item_size = os.path.getsize(item_path)
        else:
            item_size = get_size(item_path)

        if current_batch_size + item_size > THRESHOLD_SIZE:
            # Create a RAR archive for the current batch
            rar_file_path = os.path.join(directory, f"archive_{batch_index}.rar")
            rar_files(directory, current_batch, rar_file_path)
            print(f"Created {rar_file_path}")
            current_batch = []
            current_batch_size = 0
            batch_index += 1

        current_batch.append(item_path)
        current_batch_size += item_size

    # Handle the last batch
    if current_batch:
        rar_file_path = os.path.join(directory, f"archive_{batch_index}.rar")
        rar_files(directory, current_batch, rar_file_path)
        print(f"Created {rar_file_path}")

if __name__ == "__main__":
    directory = "path/to/your/directory"
    main(directory)