#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import re

# Define the directory to search
search_dir = "."

# Compile a regular expression for matching corean characters
# Covers the complete range of Hangul syllables and Jamo (modern and old)
corean_re = re.compile('[\u1100-\u11FF\u3130-\u318F\uA960-\uA97F\uD7B0-\uD7FF\uAC00-\uD7AF]')

def contains_corean(text):
    """Check if the text contains corean characters."""
    return corean_re.search(text) is not None

# Function to replace corean characters with 'x'
def replace_corean_with_x(text):
    return corean_re.sub('x', text)

def remove_files_with_corean_characters(directory):
    """Walk through the directory, check each .txt and .html file, and remove the ones with corean characters."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in ['.txt','html','js','mhtml','.md','.py','json','yaml','.xml']):
                filepath = os.path.join(root, file)
                try:
                    # Open and read the file
                    with open(filepath, 'r', encoding="utf-8") as f:
                        content = f.read()
                        has_cletter = contains_corean(content)
                    if not has_cletter: continue
                    with open(filepath, 'w', encoding="utf-8") as f:
                        modified_content = replace_corean_with_x(content)
                        # Write the modified content back to the file or a new file
                        f.write(modified_content)
                        # Delete the file if it contains corean characters
                        # os.remove(filepath)
                        print(f"Processed {filepath}")

                except UnicodeDecodeError:
                    # Handle files that cannot be decoded
                    print(f"Skipped {filepath} due to an encoding issue.")
                except Exception as e:
                    # General error handling (e.g., permission issues)
                    print(f"Error processing {filepath}: {e}")

# Call the removal function with your specified directory
remove_files_with_corean_characters(search_dir)