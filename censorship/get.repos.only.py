import os

root_dir = 'f:\\github.com'

for root, dirs, files in os.walk(root_dir):
    if root.count(os.sep) - root_dir.count(os.sep) == 1:
        for dir_name in dirs:
            if dir_name.startswith('.'): continue
            print(dir_name)