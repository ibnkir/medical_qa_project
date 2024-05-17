"""Класс FastApiHandler для обработки запросов API."""

import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import pairwise_distances
from catboost import CatBoostClassifier
from sentence_transformers import SentenceTransformer
import time


class FastApiHandler:
    """Класс FastApiHandler, который обрабатывает запросы и возвращает список похожих вопросов."""

    def __init__(self, bert_model_name="all-MiniLM-L6-v2"):
        """Инициализация переменных класса.
        Args:
            bert_model_name (str): Название используемой BERT-модели.
        """
        self.bert_model_name = bert_model_name
        
        # Типы параметров запроса для проверки
        self.param_types = {
            "user_text": str,
            "questions_num": int,
            "is_cls_needed": bool
        }
        
        self.prepare_data()
        self.encode_all_questions()
        self.load_question_pairs_classifier()

    def prepare_data(self):
        """Подготовка данных."""
        os.makedirs("./data", exist_ok=True)

        # Скачиваем датасет с парами вопросов
        try:
            # Скачиваем из локального файла предобработанный датасет, если ранее сохраняли
            self.data = pd.read_csv("../data/medical_questions_pairs.csv")
            is_preproc_needed = False
        except:
            # Скачиваем исходный датасет из Интернета
            data = load_dataset("medical_questions_pairs", split="train")
            self.data = pd.DataFrame(data)
            is_preproc_needed = True
        
        # Создаем корпус вопросов
        try:
            # Скачиваем из локального файла, если ранее сохраняли
            self.questions = pd.read_csv("../data/medical_questions.csv")
        except:
            # Создаем на основе исходного датасета
            self.questions = pd.DataFrame(pd.concat([self.data['question_1'], self.data['question_2']], axis=0).unique())
            self.questions.reset_index(inplace=True)
            self.questions.rename(columns={'index': 'question_id', 0: 'question'}, inplace=True)    
            # Сохраняем локально
            self.questions.to_csv("../data/medical_questions.csv", index=False)

        # Если пары вопросов загружены не локально, а из Интернета, то добавляем id вопросов и сохраняем локально
        if is_preproc_needed:
            # Добавляем id к вопросам из 1-й колонки
            self.data = pd.merge(self.data, self.questions, left_on='question_1', right_on='question', how='left')
            self.data.rename(columns={'question_id': 'question_1_id'}, inplace=True)
            self.data.drop(columns='question', inplace=True)
            
            # Добавляем id к вопросам из 2-й колонки
            self.data = pd.merge(self.data, self.questions, left_on='question_2', right_on='question', how='left')
            self.data.rename(columns={'question_id': 'question_2_id'}, inplace=True)
            self.data.drop(columns='question', inplace=True)

            # Сохраняем локально
            self.data.to_csv("../data/medical_questions_pairs.csv", index=False)
        
    def encode_all_questions(self):
        """Векторизация корпуса вопросов с помощью BERT-модели."""
        self.bert_model = SentenceTransformer(self.bert_model_name)
        try:
            # Скачиваем из локального файла, если ранее сохраняли
            self.bert_embeds = np.load("./data/question_bert_embeds.npy")
        except:
            self.bert_embeds = self.bert_model.encode(self.questions['question'], convert_to_tensor=False)
            with open('../data/question_bert_embeds.npy', 'wb') as f:
                np.save(f, self.bert_embeds)
            
    def load_question_pairs_classifier(self):
        """Загрузка обученного классификатора пар вопросов."""
        try:
            self.cb_cls_model = CatBoostClassifier()
            self.cb_cls_model.load_model('../models/question_pairs_cb_classifier.cbm')
        except Exception as e:
            print(f"Failed to load catboost question pairs classifier: {e}")

    def find_close_embed_inds(self, query_embed, n, is_cls_needed):
        """
        Поиск индексов n ближайших векторов.
        
        Входные данные:
            - query_embed - ndarray-вектор запрашиваемого вопроса,
            - n - количество ближайших векторов, которое нужно найти.
        
        Возвращаемое значение:
            Список из n индексов ближайших векторов.
        """
        dists = pairwise_distances(query_embed, self.bert_embeds, metric='cosine').squeeze()
        
        # Если расстояние до самого близкого вопроса меньше определенного значения, 
        # то считаем, что он совпадает с заданным вопросом и его можно игнорировать
        if dists[0] < 1e-2:
            close_embed_inds = np.argsort(dists)[1:]
        else:
            close_embed_inds = np.argsort(dists)

        n = min(n, len(close_embed_inds))
        
        if not is_cls_needed:
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
        if "user_text" not in query_params or "questions_num" not in query_params or "is_cls_needed" not in query_params:
            return False

        if not isinstance(query_params["user_text"], self.param_types["user_text"]):
            return False

        if not isinstance(query_params["questions_num"], self.param_types["questions_num"]):
            return False

        if not isinstance(query_params["is_cls_needed"], self.param_types["is_cls_needed"]):
            return False
        
        return True
          
    def handle(self, query_params):
        """Функция для обработки входящих запросов по API. 
        Запрос состоит из текста вопроса и количества похожих на него вопросов, которое нужно найти.

        Args:
            query_params (dict): Словарь параметров запроса.

        Returns:
            - **dict**: Словарь, содержащий результат выполнения запроса.
        """
        try:
            # Проверяем запрос к API
            if not self.check_query_params(query_params):
                response = {"Error": "Problem with query parameters"}
            else:
                        
                user_text = query_params["user_text"]
                questions_num = query_params["questions_num"]
                is_cls_needed = query_params["is_cls_needed"]
                
                start_time = time.time()
                query_embed = self.bert_model.encode(user_text, convert_to_tensor=False).reshape(1, -1)
                inds = self.find_close_embed_inds(query_embed, questions_num, is_cls_needed)
                similar_questions = self.questions.iloc[inds, :]['question'].values
                
                response = {
                    "similar_questions": list(similar_questions),
                    'time': time.time() - start_time
                }

        except Exception as e:
            return {"Error": "Problem with request"}
        else:
            return response
        
             
if __name__ == "__main__":

    # Создаём параметры для тестового запроса
    test_params = {
        "user_text": "How do I check my blood sugar?",
        "questions_num": 10,
        "is_cls_needed": False
    }

    # Создаём обработчик запросов для API
    handler = FastApiHandler()

    # Делаем тестовый запрос
    print(f"Searching {test_params['questions_num']} similar questions for text:\n{test_params['user_text']}\n")
    response = handler.handle(test_params)
    print(f"Response: {response}")
