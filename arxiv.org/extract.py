# -*- coding: utf-8 -*-
import re

def extract_text_in_parentheses(filename):
    # Compile a regular expression pattern to find text within parentheses
    pattern = re.compile(r'\((.*?)\)')

    try:
        # Open the file
        with open(filename, 'r',encoding="utf8") as file:
            # Read the file line by line
            for line in file:
                # Find all occurrences of text within parentheses
                matches = pattern.findall(line)
                # Print each match
                for match in matches:
                    if "https://arxiv.org" not in match: continue
                    print(match)
    except FileNotFoundError:
        print("The file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

extract_text_in_parentheses('all.txt')