"""
Joke filtering script - filtering sensitive and short jokes
This script is designed to filter the original joke data, remove jokes containing sensitive words and jokes that are not long enough.
After processing, save the valid jokes to a new file.

Workflow:
1. Read the original joke data from the CSV file and extract the joke column.
2. Load the sensitive word list from the TXT file.
3. Traverse the jokes and check whether they meet the following conditions:
    - The length is greater than 5 characters.
    - Does not contain sensitive words (strictly match by word).
4. Save the jokes that pass the filter to a new text file.

Dependencies:
- pandas
- re

Author: Jiaxin Tang, Jiayi Dong, Ximing Cai
Date: 2024/12/15
"""

import re
import pandas as pd

# Read CSV file and extract the joke column
jokes_df = pd.read_csv("data/shortjokes.csv")
jokes = jokes_df['Joke'].tolist()

# Initialize an empty sensitive word list
sensitive_words = []

# Read TXT file and add sensitive words to the list
with open("data/sensitive_words.txt", "r", encoding="utf-8") as file:
    for line in file:
        sensitive_words.append(line.strip())


def contains_sensitive_word(joke, sensitive_words):
    """
    Check if a joke contains sensitive words.

    This is done by converting the joke text to lowercase and using a regular expression to check if sensitive words appear as separate words.
    Returns True if a sensitive word exists; otherwise, returns False.

    Args:
        joke (str): Joke text to be checked.
        sensitive_words (list of str): A list of sensitive words, each of which is a string.

    Returns:
        bool: Returns True if a joke contains a sensitive word; otherwise, returns False.
    """
    joke_lower = joke.lower()
    for word in sensitive_words:
        if re.search(r'\b' + re.escape(word) + r'\b', joke_lower):
            return True
    return False


def is_valid_joke(joke, sensitive_words):
    """
    Check if a joke is valid.

    A joke is considered valid only if it meets the following conditions:
        1. It is longer than 5 characters.
        2. It does not contain sensitive words.
    Returns True if a joke is valid; otherwise, returns False.

    Args:
        joke (str): Joke text to be checked.
        sensitive_words (list of str): A list of sensitive words, each of which is a string.

    Returns:
        bool: Returns True if a joke is valid; otherwise, returns False.
    """
    if len(joke) < 5:
        return False
    if contains_sensitive_word(joke, sensitive_words):
        return False
    return True


# Filter valid jokes
filtered_jokes = [joke for joke in jokes if is_valid_joke(joke, sensitive_words)]

# Save the valid jokes into a new file
with open("data/processed_jokes.txt", "w") as f:
    for joke in filtered_jokes:
        f.write(joke.strip() + "\n")
