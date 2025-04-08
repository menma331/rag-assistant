import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from llama_index.llms.openai import OpenAI
from openai.types import ChatModel

load_dotenv()
OPENAI_PROMPT = """
You are an AI assistant designed to answer questions based solely on the provided knowledge base, integrated via Lama Index. Your task is to analyze the user's question and search for relevant chunks in the knowledge base to provide an accurate and concise answer. Follow these rules strictly:
You have been provided with information from the index in advance. You must use ONLY the retrieved chunks from the index to generate your response. Do not use any external knowledge or assumptions beyond what is explicitly provided in the chunks.

1. Use only the information from the provided knowledge base. Do not generate any information outside of it or rely on external knowledge.
2. If the question cannot be answered using the knowledge base, respond with: "There is no information on this topic."
3. Avoid assumptions, guesses, or creative embellishments. Your response must directly reflect the content of the relevant chunks.
4. Match the style and tone of your answers to the knowledge base: use clear, factual, and instructional language, similar to the documents provided.
5. Provide concise answers that address the question directly, without unnecessary elaboration.

Answer the question based on the instructions above."""


class LlamaManager:
    def __init__(
            self,
            openai_model: ChatModel,
            openai_api_key,
            openai_prompt,
            openai_temperature: float = 0.2,
            index_name: str = "happyai-rag-index",
            project_name="Default",
            organization_id="your-org-id",
            llama_api_key: str = "your-api-key"
    ):
        self.llm = OpenAI(
            api_key=openai_api_key,
            model=openai_model,
            system_prompt=openai_prompt,
            temperature=openai_temperature,
            top_p=0.9,
            max_tokens=500
        )

        self.index = LlamaCloudIndex(
            name=index_name,
            project_name=project_name,
            organization_id=organization_id,
            api_key=llama_api_key
        )

        self.query_engine = self.index.as_query_engine(llm=self.llm)

    def get_answer(self, question: str) -> dict:
        """Получение ответа на вопрос от нейросети.

        Если ответ на вопрос может быть сформирован базой данных, мы получим словарь:
        'answer': str, 'chunks': str

        Если ответ на вопрос выходит за пределы базы знаний, мы получим строку с ответом "The question is outside my knowledge base"
        """
        answer = self.query_engine.query(question)
        chunks = [node.get_content() for node in answer.source_nodes]

        # Если чанки пустые или не содержат ответа, принудительно возвращаем сообщение
        if not chunks or all(not chunk.strip() for chunk in chunks):
            return {'answer': 'The question is outside my knowledge base', 'chunks': chunks}

        result = {'answer': answer.response, 'chunks': chunks}
        return result

    def get_answers_parallel(self, questions: list[str], max_workers: int = 10) -> list[dict]:
        """Получение ответов на список вопросов параллельно с сохранением порядка."""
        answers = [None] * len(questions)  # Список для хранения ответов в нужном порядке
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Создаём задачи для каждого вопроса с сохранением их индексов
            future_to_index = {executor.submit(self.get_answer, question): index for index, question in enumerate(questions)}

            # Собираем результаты по мере завершения
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    answer = future.result()  # Получаем результат ответа
                    logging.debug(f"Answer for question {index}: {answer}")
                    answers[index] = answer  # Сохраняем ответ на соответствующий вопрос
                except Exception as e:
                    answers[index] = f"Error processing question '{questions[index]}': {str(e)}"

        return answers


llama_manager = LlamaManager(
    openai_model="gpt-4o-mini",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_prompt=OPENAI_PROMPT,
    llama_api_key=os.getenv("LLAMA_API_KEY"),
    index_name="happyai-rag-index",
    project_name="Default",
    organization_id="29ecb8ba-a016-48ed-8342-804e30b3aaaf",
    openai_temperature=0.2,
)
