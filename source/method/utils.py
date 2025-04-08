import json
import logging
import os
import nltk


def count_words(text):
    """Simple word counting."""
    return len(text.split())

def parse_pause_point(text):
    text = text.strip("Break point: ")
    if text[0] != '<':
        return None
    for i, c in enumerate(text):
        if c == '>':
            if text[1:i].isnumeric():
                return int(text[1:i])
            else:
                return None
    return None

def buildMultipleChoiceQuestionText(questionString, options):
    choicesString = ''
    for index, option in enumerate(options):
         choicesString += f'[[{index+1}]]: {option} \n'
    return f'{questionString} \nOptions: \n{choicesString}'

def buildMultipleChoiceQuestionTextWithoutNumbers(questionString, options):
    choicesString = ''
    for index, option in enumerate(options):
         choicesString += f'- {option} \n'
    return f'{questionString} \nOptions: \n{choicesString}'


def save_pages_to_json(pages, file_path):
    """
    Saves the pages (list of lists of sentences) as a JSON file.
    :param pages: List[List[str]] - A list of sentence lists (pages).
    :param file_path: str - The output JSON file path.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(pages, f, indent=4, ensure_ascii=False)  # Pretty-print for readability
        logging.info(f"Successfully saved {len(pages)} pages to {file_path}")
    except Exception as e:
        logging.error(f"Error saving pages: {e}")
        raise

def load_pages_from_json(file_path):
    """
    Loads pages from a JSON file and ensures the format is a list of lists of sentences.
    :param file_path: str - The input JSON file path.
    :return: List[List[str]] - The pages loaded from the file.
    """
    if not os.path.exists(file_path):
        logging.error(f"File {file_path} not found.")
        return None  # Or raise an exception if file must exist

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            pages = json.load(f)

        # Ensure the structure is a list of lists of strings
        if isinstance(pages, list) and all(isinstance(page, list) and all(isinstance(sentence, str) for sentence in page) for page in pages):
            logging.info(f"Successfully loaded {len(pages)} pages from {file_path}")
            return pages
        else:
            logging.error(f"Invalid file format: Expected List[List[str]], got {type(pages)}")
            return None  # Return None or raise an exception

    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON file {file_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error loading pages: {e}")
        return None


def save_shortened_pages_to_json(shortened_pages, path):
    """
    Saves shortened_pages as a JSON file.
    :param path: str - File path to save the JSON.
    """
    if shortened_pages is None:
        raise ValueError("There are no shortened pages to store.")

    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(shortened_pages, file, indent=4, ensure_ascii=False)  # Pretty JSON format
        logging.info(f"Successfully saved shortened pages to {path}")
    except Exception as e:
        logging.error(f"Error saving shortened pages: {e}")
        raise

def load_shortened_pages_from_json(path):
    """
    Loads shortened_pages from a JSON file.
    :param path: str - File path to load from.
    """
    if not isinstance(path, str):
        raise ValueError("The path must be a string.")

    if not os.path.exists(path):
        logging.error(f"File {path} not found.")
        return None  # Or raise an exception

    try:
        with open(path, "r", encoding="utf-8") as file:
            loaded_shortened_pages = json.load(file)

        # Ensure it's a dictionary
        if not isinstance(loaded_shortened_pages, list):
            raise ValueError(f"Invalid format: Expected list, got {type(loaded_shortened_pages)}")

        # Validate each value is a Node (if applicable)
        for shortened_page in loaded_shortened_pages:
            if not isinstance(shortened_page, str): 
                raise ValueError(f"A shortened pages item is not a valid Shortened Page String representation.")

        return loaded_shortened_pages
        logging.info(f"Successfully loaded shortened pages from {path}")

    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON file {path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Failed to load shortened pages from {path}: {e}")
        return None

def safe_sentence_split(text, max_words=600):
    naive_sentences = nltk.tokenize.sent_tokenize(text)
    final_sentences = []
    
    for sent in naive_sentences:
        words = sent.split()
        if len(words) <= max_words:
            final_sentences.append(sent)
        else:
            # Fallback: chunk it
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i : i + max_words])
                final_sentences.append(chunk)
    
    return final_sentences