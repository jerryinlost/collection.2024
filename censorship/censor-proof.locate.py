#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import re
import shutil

# Define the directory to search
src_dir = os.getcwd()#"."
temp_dir = os.path.abspath('../temp')
# Create the destination directory if it doesn't exist
os.makedirs(temp_dir, exist_ok=True)

# Compile a regular expression for matching corean characters
# Covers the complete range of Hangul syllables and Jamo (modern and old)
corean_re = re.compile('[\u1100-\u11FF\u3130-\u318F\uA960-\uA97F\uD7B0-\uD7FF\uAC00-\uD7AF]')

def contains_corean(text):
    """Check if the text contains corean characters."""
    return corean_re.search(text) is not None

def remove_files_with_corean_characters(directory):
    """Walk through the directory, check each .txt and .html file, and remove the ones with corean characters."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in ['.txt','html','js','mhtml','.md','.py','json','yaml','.xml']):
                filepath = os.path.join(root, file)
                try:
                    # Open and read the file
                    f = open(filepath, 'r', encoding="utf-8") 
                    content = f.read()
                    if contains_corean(content):
                        f.close()
                        # Delete the file if it contains corean characters
                        # os.remove(filepath)
                        # print(f"Removed {filepath}")

                        # Construct the relative path
                        rel_path = os.path.relpath(root, src_dir)
                        # Construct the destination path
                        dest_path = os.path.join(temp_dir, rel_path)
                        # Create the destination directory if it doesn't exist
                        os.makedirs(dest_path, exist_ok=True)
                        # Copy the file
                        shutil.copy(filepath, os.path.join(dest_path, file))
                        print(f"Copied: {filepath} to {dest_path}")                        
                    else: f.close()
                except UnicodeDecodeError:
                    # Handle files that cannot be decoded
                    print(f"Skipped {filepath} due to an encoding issue.")
                except Exception as e:
                    # General error handling (e.g., permission issues)
                    print(f"Error processing {filepath}: {e}")

# Call the removal function with your specified directory
remove_files_with_corean_characters(src_dir)