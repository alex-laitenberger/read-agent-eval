import logging

from source.method.ReadAgent import ReadAgent
from source.method.QAModels import OpenAI_QAModel_MultipleChoice
from source.method.RAModels import OpenAI_RAModel_Pagination, OpenAI_RAModel_Gisting, OpenAI_RAModel_Lookup


from source.experiments.utils import save_jsonl, log_error, create_directories, load_json_file, load_jsonl_file
from datetime import datetime
from config import OPENAI_API_KEY
import os

from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

OPENAI_MODELSTRING = "gpt-4o-mini-2024-07-18"

# Experiment metadata
EXPERIMENT_IDENTIFIER = "readagent-precreate-pages_gpt4o-mini-Quality_dev"
CURRENT_DATE_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M")

# Paths
STORED_PAGES_FOLDER_PATH = f"experiments/artifacts/pages/quality/dev/{CURRENT_DATE_TIME}-{EXPERIMENT_IDENTIFIER}"
STORED_SHORTENED_PAGES_FOLDER_PATH = f"experiments/artifacts/shortened_pages/quality/dev/{CURRENT_DATE_TIME}-{EXPERIMENT_IDENTIFIER}"
LOG_DIR = "experiments/logs/"
LOG_FILE = f"{LOG_DIR}/{CURRENT_DATE_TIME}-quality_dev_precreate_pages.log"


# Ensure necessary directories exist
create_directories([STORED_PAGES_FOLDER_PATH, STORED_SHORTENED_PAGES_FOLDER_PATH, LOG_DIR])

# Load the API key into the environment
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

 # Logging Configuration
logger = logging.getLogger()  # Get the root logger
logger.setLevel(logging.DEBUG)  # Set the general logging level for the root logger, level is set again for the handlers

# Remove existing handlers to avoid duplicates
if logger.hasHandlers():
    logger.handlers.clear()

# File handler
file_handler = logging.FileHandler(LOG_FILE, mode="w")  # Write logs to file
file_handler.setLevel(logging.INFO)  # Set log level for the file handler
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Stream handler (for terminal output)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)  # Set log level for the stream handler
stream_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(stream_formatter)
logger.addHandler(stream_handler)



def precreate_pages_for_all_docs():

    logging.info(f"Starting experiment: {EXPERIMENT_IDENTIFIER}")

    # Load preprocessed dataset
    preprocessed_path = "data/quality/preprocessed/QuALITY.v1.0.1.htmlstripped_dev_preprocessed.json"
    grouped_data = load_json_file(preprocessed_path)

    openAI_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0)

    try:
        #with ThreadPoolExecutor(max_workers=5) as executor:
        with ThreadPoolExecutor() as executor:
            logging.info("Using multithreaded Precreate_Pages")
            futures = [
                executor.submit(
                    precreate_pages_for_doc,
                    doc_id,
                    doc_data,
                    openAI_client
                )
                for doc_id, doc_data in grouped_data.items()
            ]

            for future in as_completed(futures):
                # check if a thread fails with exception
                exception = future.exception()
                if exception:
                    logging.error(f"Error in page creation for a document: {exception}")

                    # Propagate the exception
                    raise exception

    except Exception as e:
        logging.exception(f"While precreating pages the following error ocurred: {e}")

    logging.info(f"Experiment {EXPERIMENT_IDENTIFIER} completed.")



def precreate_pages_for_doc( doc_id,
                    doc_data,
                    openAI_client):
    
    try:
        # Initialize models
        logging.info("Initializing models...")
        pagination_model = OpenAI_RAModel_Pagination(modelString=OPENAI_MODELSTRING, client=openAI_client)
        gisting_model = OpenAI_RAModel_Gisting(modelString=OPENAI_MODELSTRING, client=openAI_client)
        lookup_model = OpenAI_RAModel_Lookup(modelString=OPENAI_MODELSTRING, client=openAI_client)
        qa_model = OpenAI_QAModel_MultipleChoice(modelString=OPENAI_MODELSTRING, client=openAI_client)

        # Initialize ReadAgent
        readAgent = ReadAgent(pagination_model, gisting_model, lookup_model, qa_model)

        logging.info(f"Processing document {doc_id}...")

        # Extract document context
        document_context = doc_data['article']

        readAgent.create_pages(document_context)
        readAgent.save_pages(f"{STORED_PAGES_FOLDER_PATH}/{doc_id}.json")
        
        readAgent.shorten_pages()
        readAgent.save_shortened_pages(f"{STORED_SHORTENED_PAGES_FOLDER_PATH}/{doc_id}.json")

        logging.info(f"Finished creating pages and shortened_pages for document {doc_id}.")

    except Exception as e:        
        logging.exception(f"Error precreating pages for doc {doc_id}")
        raise e

if __name__ == "__main__":
    precreate_pages_for_all_docs()
