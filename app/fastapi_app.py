"""FastAPI-приложение для поиска похожих вопросов.

Для запуска перейти в папку app и выполнить команду:
uvicorn fastapi_app:app --reload --port 8081 --host 0.0.0.0

либо, если работа ведется полностью локально:
uvicorn fastapi_app:app --reload --port 8081 --host 127.0.0.1

Если используется другой порт, то заменить 8081 на этот порт.

Для просмотра документации API и совершения тестовых запросов через 
Swagger UI зайти на  http://127.0.0.1:8081/docs

либо запустить Streamlit-приложение (см. streamlit_app.py)
"""

import uvicorn
from fastapi import FastAPI
from fastapi_handler import FastApiHandler

# Создаём приложение FastAPI
app = FastAPI()

# Создаём обработчик запросов для API
app.handler = FastApiHandler()

@app.get("/")
def read_root():
    return {'message': 'Welcome from the API'}

@app.post("/similar_questions/") 
def find_similar_questions(user_text: str, questions_num: int, cls_option: str) -> dict:
    """Поиск медицинских вопросов, похожих на введенный пользователем текст.
    Args:
        - user_text (str): Текст пользователя,
        - questions_num (int): Количество похожих вопросов, которое нужно найти,
        - cls_option (str): Название классификатора либо 'None'.

    Returns:
        - dict.
    """
    query_params = {
        "user_text": user_text,
        "questions_num": questions_num,
        'cls_option': cls_option
    }
    
    return app.handler.handle(query_params)


if __name__ == "__main__":
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port="8081")