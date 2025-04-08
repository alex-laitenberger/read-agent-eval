import os
import logging
import tiktoken

from tenacity import retry, stop_after_attempt, wait_exponential, after_log, before_sleep_log

logger = logging.getLogger(__name__)

tokenizer = tiktoken.get_encoding("cl100k_base")

class OpenAI_RAModel_Pagination():
    def __init__(self, modelString, client):
        """
        Initializes the OpenAI model with the model name set in the modelString

        Args:
            modelName (str): The OpenAI model.
        """
        self.modelString = modelString
        self.client = client

    @retry(wait=wait_exponential(multiplier=1, max=60), 
        stop=stop_after_attempt(10), 
        before_sleep=before_sleep_log(logger, logging.INFO), 
        after=after_log(logger, logging.INFO), 
        reraise=True)
    def paginate(
        self, preceding_text, passage_text, end_tag, max_decode_steps: int = 512
    ):
        """
        Generates Answers to specified multiple choice questions and options optimized for QuALITY benchmark.
        """
        pagination_prompt = f"""
You are given a passage that is taken from a larger meeting transcript.
There are some numbered labels between the paragraphs (like <0>) in the passage.
Please choose one label at a natural transition in the passage.
For example, the label can be at the end of a dialogue, the end of an argument, a change in the topic being discussed, etc.
Please respond with the label and explain your choice.
For example, if <57> is a natural transition, answer with "Label: <57>\n Because ..."

Passage:

{preceding_text}
{passage_text}
{end_tag}

"""
# preceding_text: a fraction of previous context
# passage_text: a chunk of text.
# end_tag: a string, whose value is "" if the text is at the end of the article, and otherwise "\n...".


        promptLog = f"\n\n#### Prompting {self.modelString}: ####\n\n{pagination_prompt}\n\n#### End of Prompt ####\n\n"
        logging.debug(promptLog)

        raw_response = self.client.chat.completions.with_raw_response.create(
            model=self.modelString,
            max_tokens=max_decode_steps,
            temperature=0,
            seed = 42,
            messages=[
              {'role': 'user', 'content': pagination_prompt},
            ]
          )

        completion = raw_response.parse()    
        answerString = completion.choices[0].message.content.strip()
        
        answerLog = f"\n\n#### {self.modelString} Response: ####\n\n{answerString}\n\n#### End of Response ####\n\n"
        logging.debug(answerLog)
        
        return answerString

class OpenAI_RAModel_Gisting():
    def __init__(self, modelString, client):
        """
        Initializes the OpenAI model with the model name set in the modelString

        Args:
            modelName (str): The OpenAI model.
        """
        self.modelString = modelString
        self.client = client

    @retry(wait=wait_exponential(multiplier=1, max=60), 
        stop=stop_after_attempt(10), 
        before_sleep=before_sleep_log(logger, logging.INFO), 
        after=after_log(logger, logging.INFO), 
        reraise=True)
    def shorten_page(
        self, page, max_decode_steps: int = 512
    ):
        """
        Generates Answers to specified multiple choice questions and options optimized for QuALITY benchmark.
        """
        shorten_prompt = f"""
Please shorten the following passage.
Just give me a shortened version. DO NOT explain your reason.

Passage:
{page}

"""
# preceding_text: a fraction of previous context
# passage_text: a chunk of text.
# end_tag: a string, whose value is "" if the text is at the end of the article, and otherwise "\n...".


        promptLog = f"\n\n#### Prompting {self.modelString}: ####\n\n{shorten_prompt}\n\n#### End of Prompt ####\n\n"
        logging.debug(promptLog)

        raw_response = self.client.chat.completions.with_raw_response.create(
            model=self.modelString,
            max_tokens=max_decode_steps,
            temperature=0,
            seed = 42,
            messages=[
              {'role': 'user', 'content': shorten_prompt},
            ]
          )

        completion = raw_response.parse()    
        answerString = completion.choices[0].message.content.strip()
        
        answerLog = f"\n\n#### {self.modelString} Response: ####\n\n{answerString}\n\n#### End of Response ####\n\n"
        logging.debug(answerLog)
        
        return answerString

class OpenAI_RAModel_Lookup():
    def __init__(self, modelString, client):
        """
        Initializes the OpenAI model with the model name set in the modelString

        Args:
            modelName (str): The OpenAI model.
        """
        self.modelString = modelString
        self.client = client

    @retry(wait=wait_exponential(multiplier=1, max=60), 
        stop=stop_after_attempt(10), 
        before_sleep=before_sleep_log(logger, logging.INFO), 
        after=after_log(logger, logging.INFO), 
        reraise=True)
    def lookup(
        self, shortened_article, question, max_lookup_pages, max_decode_steps: int = 512
    ):
        """
        Generates Answers to specified multiple choice questions and options optimized for QuALITY benchmark.
        """
        lookup_prompt = f"""
The following text is what you remembered from reading an article and a multiple choice question related to it.
You may read 1 to {max_lookup_pages} page(s) of the article again to refresh your memory to prepare yourselve for the question.
Please respond with which page(s) you would like to read.
For example, if your only need to read Page 8, respond with \"I want to look up Page [8] to ...\";
if your would like to read Page 7 and 12, respond with \"I want to look up Page [7, 12] to ...\";
if your would like to read Page 2, 3, 7, 15 and 18, respond with \"I want to look up Page [2, 3, 7, 15, 18] to ...\".
if your would like to read Page 3, 4, 5, 12, 13 and 16, respond with \"I want to look up Page [3, 3, 4, 12, 13, 16] to ...\".
DO NOT select more pages if you don't need to.
DO NOT answer the question yet.

Text:
{shortened_article}

Question:
{question}

Take a deep breath and tell me: Which 1 to {max_lookup_pages} page(s) would you like to read again?

"""
# preceding_text: a fraction of previous context
# passage_text: a chunk of text.
# end_tag: a string, whose value is "" if the text is at the end of the article, and otherwise "\n...".

        used_input_tokens = len(tokenizer.encode(lookup_prompt))

        promptLog = f"\n\n#### Prompting {self.modelString}: ####\n\n{lookup_prompt}\n\n#### End of Prompt ####\n\n"
        logging.debug(promptLog)

        raw_response = self.client.chat.completions.with_raw_response.create(
            model=self.modelString,
            max_tokens=max_decode_steps,
            temperature=0,
            seed = 42,
            messages=[
              {'role': 'user', 'content': lookup_prompt},
            ]
          )

        completion = raw_response.parse()    
        answerString = completion.choices[0].message.content.strip()
        
        answerLog = f"\n\n#### {self.modelString} Response: ####\n\n{answerString}\n\n#### End of Response ####\n\n"
        logging.debug(answerLog)
        
        return answerString, used_input_tokens