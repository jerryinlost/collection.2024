import os
import shutil

remove_list = ["refs","snapshots",".noexist"]

def remove_refs_folders(root_dir):
    """
    Recursively remove subfolders named 'refs' in the specified root directory.
    
    :param root_dir: The root directory to start the search.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            if any(dirname == item for item in remove_list):
                refs_path = os.path.join(dirpath, dirname)
                print(f"Removing: {refs_path}")
                shutil.rmtree(refs_path)

# Example usage
root_directory = 'C:\\Users\\kojy\\.cache\\huggingface\\hub'
remove_refs_folders(root_directory)