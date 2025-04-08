import logging
from source.method.ReadAgent import ReadAgent
from source.method.QAModels import OpenAI_QAModel_MultipleChoice
from source.method.RAModels import OpenAI_RAModel_Pagination, OpenAI_RAModel_Gisting, OpenAI_RAModel_Lookup

from source.experiments.utils import save_jsonl, log_error, create_directories, load_json_file, load_jsonl_file, extract_number
from datetime import datetime
from config import OPENAI_API_KEY
import os

from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI


# Constant Paths for precreated pages and shortened pages
STORED_PAGES_FOLDER_PATH = "experiments/artifacts/pages/infinity_bench/longbook_choice_eng/2025-04-08_13-13-readagent-precreate-pages-gpt4o-mini"
STORED_SHORTENED_PAGES_FOLDER_PATH = "experiments/artifacts/shortened_pages/infinity_bench/longbook_choice_eng/2025-04-08_13-13-readagent-precreate-pages-gpt4o-mini"

# Parameters
#OPENAI_MODELSTRING = "gpt-4o-2024-11-20"
OPENAI_MODELSTRING = "gpt-4o-mini-2024-07-18"

# Load the API key into the environment
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def run_experiment_for_all_docs(experiment_identifier, hyperparams):
    current_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    stored_answers_path = f"experiments/artifacts/answers/infinity_bench/longbook_choice_eng"
    stored_answers_file = f"{stored_answers_path}/{current_date_time}-{experiment_identifier}.jsonl"
    stored_errors_file = f"{stored_answers_path}/{current_date_time}-{experiment_identifier}_ERRORS.jsonl"
    log_dir = "experiments/logs/"
    log_file = f"{log_dir}/{current_date_time}-infinity_bench_longbook_choice_eng_run_experiment_{experiment_identifier}.log"

    # Ensure necessary directories exist
    create_directories([stored_answers_path, log_dir])

    # Logging Configuration
    logger = logging.getLogger()  # Get the root logger
    logger.setLevel(logging.DEBUG)  # Set the general logging level for the root logger, level is set again for the handlers
    
    # Remove existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file, mode="w")  # Write logs to file
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

    logging.info(f"Starting experiment: {experiment_identifier}")

    # Load preprocessed dataset
    preprocessed_path = "data/infinity_bench/preprocessed/longbook_choice_eng_preprocessed.json"
    grouped_data = load_json_file(preprocessed_path)

    openAI_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0)

    try:
        #with ThreadPoolExecutor(max_workers=1) as executor: #optionally control amount of parallelity
        with ThreadPoolExecutor() as executor:
            logging.info("Using multithreaded Precreate_Pages")
            futures = [
                executor.submit(
                    run_experiment_for_doc,
                    doc_id,
                    doc_data,
                    openAI_client,
                    hyperparams,
                    stored_answers_file, 
                    stored_errors_file
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
        logging.exception(f"While running experiments the following error ocurred: {e}")

    logging.info(f"Experiment {experiment_identifier} completed.")

def run_experiment_for_doc(doc_id, doc_data, openAI_client, hyperparams, stored_answers_file, stored_errors_file):
    
    if doc_id == "34e7b2fa12fdd1206e0e8fe3bb82468d":
        logging.info("Skipping document 34e7b2fa12fdd1206e0e8fe3bb82468d, being too big for context size")
        return

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

        # Load precreated pages and shortened pages
        readAgent.load_pages(f"{STORED_PAGES_FOLDER_PATH}/{doc_id}.json")    
        readAgent.load_shortened_pages(f"{STORED_SHORTENED_PAGES_FOLDER_PATH}/{doc_id}.json")
        logging.info(f"Loaded precreated pages and shortened_pages for document {doc_id}.")

         # Iterate over questions in the document
        for entry in doc_data["entries"]:
            question_id = entry["question_id"]
            question = entry["input"]
            options = entry["options"]
            gold_choice = entry["gold_choice"]

            # Answer the question

            answer, looked_up_page_ids, used_input_tokens = readAgent.answer_question(
                question=question,
                options=options,
                max_lookup_pages=hyperparams["max_lookup_pages"]
            )

            if isinstance(answer, str):
                predicted_choice = extract_number(answer)
                correct_choice = predicted_choice == gold_choice
                logging.info(
                    f"Question ID: {question_id}, Predicted Choice: {predicted_choice}, Correct: {correct_choice}"
                )

                # Store the answer
                result = {
                    "document_id": doc_id,
                    "question_id": question_id,
                    "gold": gold_choice,
                    "predicted_choice": predicted_choice,
                    "correct_choice": correct_choice,
                    "predicted_answer": answer.replace("\n", " "),
                    "looked_up_page_ids": looked_up_page_ids,
                    "used_tokens": used_input_tokens,
                }
                save_jsonl(result, stored_answers_file)
                
            else:
                log_error(doc_id, question_id, "No valid string answer", stored_errors_file)

    except Exception as e:        
        logging.exception(f"Error running experiment for doc {doc_id}")
        raise e

def run_experiment_batch():
    """Run a batch of experiments with varying configurations."""

    experiment_tag = "read-agent"

    experiments = [
        {"max_lookup_pages": 6}
        ]

    for index, hyperparams in enumerate(experiments):
        experiment_identifier = f"{experiment_tag}_{index}_m-lu-pages-{hyperparams['max_lookup_pages']}_{OPENAI_MODELSTRING}"
        run_experiment_for_all_docs(experiment_identifier, hyperparams)

if __name__ == "__main__":
    run_experiment_batch()

