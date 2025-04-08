import json
import logging
import time
from typing import List, Dict, Any

import gspread
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import FactualCorrectness, NoiseSensitivity

from llama import llama_manager
from google_sheet import sheet

# Constants
CONFIG = {
    "FILE_PATH": "creds/assistant_file_updated_questions.xlsx",
    "SHEET_NAME": 0,
    "QUESTION_COL": 2,
    "GROUND_TRUTH_COL": 3,
    "LLM_MODEL": "gpt-4o-mini",
    "MAX_WORKERS": 6,
    "TIMEOUT": 290,
    "RETRY_DELAY": 30,
    "REQUEST_DELAY": 1
}

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(file_path: str, sheet_name: int) -> tuple[List[str], List[str]]:
    """Загрузка вопросов и правильных ответов из Excel файлов."""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        questions = df.iloc[:, CONFIG["QUESTION_COL"]].tolist()
        ground_truth = df.iloc[:, CONFIG["GROUND_TRUTH_COL"]].tolist()
        logger.info("Successfully loaded questions and ground truth")
        return questions, ground_truth
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def prepare_dataset(questions: List[str], ground_truth: List[str]) -> Dataset:
    """Подготовка dataset к evaluation."""
    results = llama_manager.get_answers_parallel(
        questions,
        max_workers=CONFIG["MAX_WORKERS"]
    )

    data_dict = {
        "user_input": questions,
        "retrieved_contexts": [result['chunks'] for result in results],
        "response": [result['answer'] for result in results],
        "reference": ground_truth
    }

    return Dataset.from_dict(data_dict)


def evaluate_dataset(dataset: Dataset) -> Any:
    """Evaluate dataset через RAGAS метрики."""
    rag_llm = ChatOpenAI(model=CONFIG["LLM_MODEL"])
    evaluator_llm = LangchainLLMWrapper(langchain_llm=rag_llm)
    eval_run_config = RunConfig(timeout=CONFIG["TIMEOUT"])

    return evaluate(
        dataset=dataset,
        metrics=[FactualCorrectness(), NoiseSensitivity()],
        llm=evaluator_llm,
        run_config=eval_run_config
    )


def update_google_sheet(results: List[Dict], evaluation_result: Any) -> None:
    """Обновление Google Sheet с результатами и метриками."""
    for i, (result, metrics_result) in enumerate(zip(results, evaluation_result.traces)):
        logger.info(f'Processing record #{i + 1}')
        answer = result['answer']
        chunks = ''.join(result.get('chunks', []))
        metrics = json.dumps(metrics_result.scores)

        try:
            sheet.update_cell(row=i + 2, col=2, value=answer)
            sheet.update_cell(row=i + 2, col=3, value=chunks)
            sheet.update_cell(row=i + 2, col=4, value=metrics)
        except gspread.exceptions.APIError:
            logger.warning(f"API rate limit exceeded. Waiting {CONFIG['RETRY_DELAY']} seconds...")
            time.sleep(CONFIG["RETRY_DELAY"])
            continue

        time.sleep(CONFIG["REQUEST_DELAY"])


def main():
    load_dotenv()

    try:
        # Load and prepare data
        questions, ground_truth = load_data(
            CONFIG["FILE_PATH"],
            CONFIG["SHEET_NAME"]
        )
        dataset = prepare_dataset(questions, ground_truth)

        # Evaluate responses
        evaluation_result = evaluate_dataset(dataset)

        # Update Google Sheet
        results = llama_manager.get_answers_parallel(
            questions,
            max_workers=CONFIG["MAX_WORKERS"]
        )
        update_google_sheet(results, evaluation_result)

        logger.info("Process completed successfully")
    except Exception as e:
        logger.error(f"Process failed: {e}")
        raise


if __name__ == "__main__":
    main()