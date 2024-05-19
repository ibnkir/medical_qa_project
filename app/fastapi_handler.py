"""Класс FastApiHandler для обработки запросов API.

Для тестирования перейти в папку app и выполнить команду
python -m fastapi_handler
"""

import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import pairwise_distances
from catboost import CatBoostClassifier
from sentence_transformers import SentenceTransformer
import time
from sentence_pair_classifier import SentencePairClassifier
import torch


class FastApiHandler:
    """Класс FastApiHandler, обрабатывает запросы и возвращает список похожих вопросов."""

    def __init__(self, bert_model_name="all-MiniLM-L6-v2"):
        """Инициализация переменных класса.
        Args:
            bert_model_name (str): Название используемой BERT-модели для векторизации текста.
        """
        self.bert_model_name = bert_model_name
        
        # Типы параметров запроса для проверки
        self.param_types = {
            "user_text": str,
            "questions_num": int,
            "cls_option": str
        }
        
        self.load_data()
        self.encode_all_questions()
        self.load_classifiers()

    def load_data(self):
        """Загрузка данных."""
        os.makedirs("../data", exist_ok=True)

        # Загружаем исходный датасет с парами вопросов
        try:
            # Скачиваем из локального файла, если ранее сохраняли
            self.data = pd.read_csv("../data/medical_questions_pairs.csv")
        except:
            # Скачиваем из Интернета
            data = load_dataset("medical_questions_pairs", split="train")
            self.data = pd.DataFrame(data)
            self.data.to_csv("../data/medical_questions_pairs.csv", index=False)

        # Создаем корпус вопросов
        try:
            # Скачиваем из локального файла, если ранее сохраняли
            self.questions = pd.read_csv("../data/medical_questions.csv")
        except:
            # Создаем на основе исходного датасета
            self.questions = pd.DataFrame(pd.concat([self.data['question_1'], self.data['question_2']], axis=0).unique())
            self.questions.reset_index(inplace=True)
            self.questions.rename(columns={'index': 'question_id', 0: 'question'}, inplace=True)    
            self.questions.to_csv("../data/medical_questions.csv", index=False)

    def encode_all_questions(self):
        """Векторизация корпуса вопросов с помощью BERT-модели."""
        self.bert_model = SentenceTransformer(self.bert_model_name)
        try:
            # Скачиваем из локального файла, если ранее сохраняли
            self.bert_embeds = np.load("../data/question_bert_embeds.npy")
        except:
            self.bert_embeds = self.bert_model.encode(self.questions['question'], convert_to_tensor=False)
            with open('../data/question_bert_embeds.npy', 'wb') as f:
                np.save(f, self.bert_embeds)
            
    def load_classifiers(self):
        """Загрузка обученных классификаторов пар вопросов."""
        
        # Пробуем загрузить CatBoost
        try:
            self.cb_cls_model = CatBoostClassifier()
            self.cb_cls_model.load_model('../models/question_pairs_cb_classifier.cbm')
        except Exception as e:
            self.cb_cls_model = None
            print(f"Failed to load catboost question pairs classifier: {e}")

        # Пробуем загрузить BERT
        try:
            self.bert_cls_nn = SentencePairClassifier()
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.bert_cls_nn.load_state_dict(torch.load('../models/question_pairs_bert_classifier.pt', map_location=self.device))
        except Exception as e:
            self.bert_cls_nn = None
            print(f"Failed to load bert question pairs classifier: {e}")

    def find_similar_questions_inds(self, user_text, n, cls_option):
        """
        Поиск индексов n ближайших векторов.
        
        Args:
            - user_text (str): введенный пользователем текст,
            - n (int): количество ближайших векторов, которое нужно найти.
        
        Returns:
            Список из n индексов ближайших векторов.
        """
        query_embed = self.bert_model.encode(user_text, convert_to_tensor=False).reshape(1, -1)
        dists = pairwise_distances(query_embed, self.bert_embeds, metric='cosine').squeeze()
        
        # Если расстояние до самого близкого вопроса меньше определенного значения, 
        # то считаем, что он совпадает с заданным вопросом и его можно игнорировать
        if dists[0] < 1e-2:
            close_embed_inds = np.argsort(dists)[1:]
        else:
            close_embed_inds = np.argsort(dists)

        n = min(n, len(close_embed_inds))
        
        if cls_option == 'None':
            return close_embed_inds[:n]
            
        res_inds = []
        inds_labeled_0 = []
        cnt = 0
                
        for idx in close_embed_inds:
            close_embed = self.bert_embeds[idx].reshape(1, -1)
            
            left_pair = np.hstack([query_embed, close_embed])
            if self.cls_model.predict(left_pair)[0] == 1:
                res_inds.append(idx)
                cnt += 1
            else:
                right_pair = np.hstack([close_embed, query_embed])
                if self.cls_model.predict(right_pair)[0] == 1:
                    res_inds.append(idx)
                    cnt += 1
                else:
                    inds_labeled_0.append(idx)
            
            if cnt == n:
                break
    
        # Если количество найденных индексов оказалось меньше n, то добавляем индексы ближайших векторов, 
        # для которых классификатор выдал метку 0
        for i in range(n - cnt):
            res_inds.append(inds_labeled_0[i])
            
        return res_inds
    
    def check_query_params(self, query_params: dict) -> bool:
        """Проверяем параметры запроса на наличие обязательного набора.

        Args:
            query_params (dict): Параметры запроса.

        Returns:
            bool: True — если есть нужные параметры, False — иначе
        """
        if "user_text" not in query_params or "questions_num" not in query_params or "cls_option" not in query_params:
            return False

        if not isinstance(query_params["user_text"], self.param_types["user_text"]):
            return False

        if not isinstance(query_params["questions_num"], self.param_types["questions_num"]):
            return False

        if not isinstance(query_params["cls_option"], self.param_types["cls_option"]):
            return False
        
        return True
          
    def handle(self, query_params):
        """Функция для обработки входящих запросов по API. 
        Запрос состоит из текста вопроса и количества похожих на него вопросов, которое нужно найти.

        Args:
            - query_params (dict): Словарь параметров запроса.

        Returns:
            - **dict**: Словарь, содержащий результат выполнения запроса.
        """
        try:
            # Проверяем параметры запроса
            if not self.check_query_params(query_params):
                response = {
                    "status": "Error",
                    "message": "Problem with query parameters."
                }
            else:   
                user_text = query_params["user_text"]
                questions_num = query_params["questions_num"]
                cls_option = query_params["cls_option"]

                # Если пользователь выбрал классификатор, который незагружен
                if cls_option == 'CatBoostClassifier' and self.cb_cls_model == None:
                    message = 'CatBoostClassifier not found, query completed without classifier.'
                    cls_option = 'None'
                elif cls_option == 'Fine-tuned BERT' and self.bert_cls_nn == None:
                    message = 'Fine-tuned BERT not found, query completed without classifier.'
                    cls_option = 'None'
                else:
                    message = ''

                start_time = time.time()
                inds = self.find_similar_questions_inds(user_text, questions_num, cls_option)
                similar_questions = self.questions.iloc[inds, :]['question'].values
                
                response = {
                    'status': 'OK',
                    "message": message,
                    "similar_questions": list(similar_questions),
                    'time': time.time() - start_time
                }

        except Exception as e:
            return {
                "status": "Error",
                "message": f"Problem with request {e}"
            }
        else:
            return response
        
             
if __name__ == "__main__":

    # Создаём параметры для тестового запроса
    test_params = {
        "user_text": "How do I check my blood sugar?",
        "questions_num": 10,
        "cls_option": "No classifier"
    }

    # Создаём обработчик запросов для API
    handler = FastApiHandler()

    # Делаем тестовый запрос
    print(f"Searching {test_params['questions_num']} similar questions for text:\n{test_params['user_text']}\n")
    response = handler.handle(test_params)
    print(f"Response: {response}")