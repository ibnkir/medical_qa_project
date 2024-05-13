"""FastAPI-приложение для поиска похожих медицинских вопросов.

Пример запуска из директории medical_qa_project/app:
uvicorn fastapi_app:app --reload --port 8081 --host 0.0.0.0

Для просмотра документации API и совершения тестовых запросов зайти на  http://127.0.0.1:8081/docs

Если используется другой порт, то заменить 8081 на этот порт
"""

from fastapi import FastAPI, Body
from fastapi_handler import FastApiHandler


# создаём приложение FastAPI
app = FastAPI()

# создаём обработчик запросов для API
app.handler = FastApiHandler()

@app.post("/similar_questions/") 
def find_similar_questions(user_text: str, questions_num: int):
    """Поиск медицинских вопросов, похожих на введенный пользователем текст.

    Args:
        user_text (str): Текст пользователя.
        questions_num (int): Количество похожих вопросов, которое нужно найти.

    Returns:
        dict: Словарь с полем 'similar_questions', которое содержит список похожих вопросов.
    """
    query_params = {
        "user_text": user_text,
        "questions_num": questions_num
    }
    return app.handler.handle(query_params)
