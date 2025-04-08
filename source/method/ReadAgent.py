# Portions of this file are derived from ReadAgent by Kuang-Huei Lee
# (https://read-agent.github.io/)
# No license was specified in the original source.
# This file is excluded from the MIT license covering the rest of the repository.
# Users are responsible for checking the original licensing terms before reuse.


import json

import logging
import os
from .QAModels import BaseQAModel
from source.method.utils import (count_words, parse_pause_point, save_pages_to_json, load_pages_from_json, save_shortened_pages_to_json, load_shortened_pages_from_json, buildMultipleChoiceQuestionText, buildMultipleChoiceQuestionTextWithoutNumbers, safe_sentence_split)

#only for testing, later delete?
from .QAModels import OpenAI_QAModel_MultipleChoice
from .RAModels import OpenAI_RAModel_Pagination, OpenAI_RAModel_Gisting, OpenAI_RAModel_Lookup

from openai import OpenAI
from config import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

import pickle

import nltk
# Ensure the 'punkt' dataset is available
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading 'punkt' dataset...")
    nltk.download('punkt_tab')

class ReadAgent:
    def __init__(self, pagination_model, gisting_model, lookup_model, qa_model):    
        self.pages = []
        self.shortened_pages = []
        self.shortened_article = ""
        self.pagination_model = pagination_model
        self.gisting_model = gisting_model
        self.lookup_model = lookup_model
        self.qa_model = qa_model

    def create_pages(   self, 
                        text: str,
                        word_limit=600,
                        start_threshold=280,
                        max_retires=10,
                        min_words_to_start_pagination = 350,
                        allow_fallback_to_last=True 
                    ):

        #using nltk sentences since datasets do not safely split paragraphs at \n
        sentences = safe_sentence_split(text, word_limit)

        logging.info(f"Split document into {len(sentences)} sentences.")

        i = 0
        pages = []
        while i < len(sentences):
            preceding = "" if i == 0 else "...\n" + '\n'.join(pages[-1])
            passage = [sentences[i]]
            wcount = count_words(sentences[i])
            j = i + 1
            while wcount < word_limit and j < len(sentences):
                wcount += count_words(sentences[j])
                if wcount >= start_threshold:
                    passage.append(f"<{j}>")
                passage.append(sentences[j])
                j += 1
            passage.append(f"<{j}>")
            end_tag = "" if j == len(sentences) else sentences[j] + "\n..."

            pause_point = None
            if wcount < min_words_to_start_pagination:
                pause_point = len(sentences)
            else:
                response = self.pagination_model.paginate(preceding, '\n'.join(passage), end_tag)
                
                pause_point = parse_pause_point(response)

                if pause_point and (pause_point <= i or pause_point > j):
                    logging.info(f"passage:\n{passage},\nresponse:\n{response}\n")
                    logging.info(f"i:{i} j:{j} pause_point:{pause_point}")
                    pause_point = None
                if pause_point is None:
                    if allow_fallback_to_last:
                        pause_point = j
                    else:
                        raise ValueError(f"preceding: {preceding}, passage: {passage}, end_tag: {end_tag}, \n\nresponse: {response}\n")

            page = sentences[i:pause_point]
            pages.append(page)            
            logging.debug(f"Paragraph {i}-{pause_point-1}: {page}")
            i = pause_point
        logging.info(f"[Pagination] Done with {len(pages)} pages")

        self.pages = pages
        
        return pages

    def shorten_pages(self):
        
        if not self.pages:  # Checks if list is empty
            raise ValueError("Error: The pages array is empty.")

        shortened_pages = []
        for i, page in enumerate(self.pages):
            shortened_text = self.gisting_model.shorten_page('\n'.join(page))
            
            shortened_pages.append(shortened_text)
            logging.debug(f"[gist] page {i}: {shortened_text}")
        
        self.shortened_pages = shortened_pages
        logging.info(f"[Gisting] Shortened {len(shortened_pages)} pages.")

        return shortened_pages


    def save_pages(self, path):
        save_pages_to_json(self.pages, path)

    def load_pages(self, path):
        self.pages = load_pages_from_json(path)

    def save_shortened_pages(self, path):
        save_shortened_pages_to_json(self.shortened_pages, path)

    def load_shortened_pages(self, path):
        self.shortened_pages = load_shortened_pages_from_json(path)


    def answer_question(self,
        question,
        options = None, #in case of multiple-choice
        max_lookup_pages = 6
        ):

        #for MC baking the options into the retrievalQuestion:
        lookupQuestion = question
        if options:
            lookupQuestion = buildMultipleChoiceQuestionTextWithoutNumbers(question, options)

        #lookup prompt:
        model_choices = []
        lookup_page_ids = []
        shortened_pages_pidx = []

        for i, shortened_text in enumerate(self.shortened_pages):
            shortened_pages_pidx.append("<Page {}>\n".format(i) + shortened_text)
        shortened_article = '\n'.join(shortened_pages_pidx)

        expanded_gist_word_counts = []
        page_ids = []


        response, lookup_used_input_tokens = self.lookup_model.lookup(shortened_article, lookupQuestion, max_lookup_pages)

        try: start = response.index('[')
        except ValueError: start = len(response)
        try: end = response.index(']')
        except ValueError: end = 0
        if start < end:
            page_ids_str = response[start+1:end].split(',')
            page_ids = []
            for p in page_ids_str:
                if p.strip().isnumeric():
                    page_id = int(p)
                    if page_id < 0 or page_id >= len(self.pages):
                        print("Skip invalid page number: ", page_id, flush=True)
                    else:
                        page_ids.append(page_id)

        logging.info(f"Model chose to look up page {page_ids}")

        # Memory expansion after look-up, replacing the target shortened page with the original page
        expanded_shortened_pages = self.shortened_pages[:]
        if len(page_ids) > 0:
            for page_id in page_ids:
                expanded_shortened_pages[page_id] = '\n'.join(self.pages[page_id])

        expanded_shortened_article = '\n'.join(expanded_shortened_pages)
        logging.debug(f"Expanded shortened article: \n{expanded_shortened_article}")

        #prompt_answer = prompt_answer_template.format(expanded_shortened_article, q, '\n'.join(options_i))
        answerString, qa_used_input_tokens = self.qa_model.answer_question(expanded_shortened_article, question, options)

        used_total_input_tokens = qa_used_input_tokens + lookup_used_input_tokens

        return answerString, page_ids, used_total_input_tokens



