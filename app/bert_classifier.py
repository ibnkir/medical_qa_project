"""Класс SentencePairClassifier и вспомогательные функции для работы с 
классификатором на основе дообученной BERT-модели с добавленным линейным слоем.
"""

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# Инициализируем константы
BERT_MODEL_NAME = 'bert-base-cased'
MAX_LEN = 256 # Максимальная общая длина двух вопросов (в токенах)
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 4
THRESHOLD = 0.5 # Порог вероятности для предсказания меток


class SentencePairClassifier(nn.Module):
    """Класс SentencePairClassifier - нейросеть из BERT-модели и линейного слоя для классификации."""

    def __init__(self, bert_model=BERT_MODEL_NAME, freeze_bert=False):
        super(SentencePairClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model)

        if freeze_bert:
            # Обучаем веса только у классификатора
            for p in self.bert.parameters():
                p.requires_grad = False

        self.dropout = nn.Dropout(p=0.1) # Слой для регуляризации
        self.cls = nn.Linear(self.bert.config.hidden_size, 1) # Слой для классификации

    def forward(self, input_ids, attn_masks, token_type_ids):
        bert_output = self.bert(input_ids, attn_masks, token_type_ids)
        pooler_output = bert_output['pooler_output']
        logits = self.cls(self.dropout(pooler_output))
        return logits
    

class CustomDataset(Dataset):
    """Класс CustomDataset для итерирования по закодированным с помощью BERT данным."""

    def __init__(self, data,  with_labels=True, max_len=MAX_LEN, bert_model=BERT_MODEL_NAME):
        super().__init__()
        self.data = data  # Pandas Dataframe
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model) 
        self.max_len = max_len # Максимальная суммарная длина двух текстов
        self.with_labels = with_labels # True для обучения, False для инференса

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        q1 = self.data.loc[index, 'question_1']
        q2 = self.data.loc[index, 'question_2']

        encoded_pair = self.tokenizer(q1, q2,
                                      padding='max_length',
                                      truncation=True,
                                      max_length=self.max_len,
                                      return_tensors='pt')

        input_ids = encoded_pair['input_ids'].squeeze(0)
        attn_masks = encoded_pair['attention_mask'].squeeze(0)
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0) # "0" для токенов 1-го вопроса, "1" для 2-го

        if self.with_labels:
            label = self.data.loc[index, 'label']
            return input_ids, attn_masks, token_type_ids, torch.tensor(label, dtype=torch.long)
        else:
            return input_ids, attn_masks, token_type_ids
        

def get_bert_preds(net, device, data_loader, threshold=THRESHOLD):
    """
    Функция для получения предсказаний классификатора на основе 
    дообученного BERT с добавленным линейным слоем.
    """
    net.eval()

    res_preds = []
    with torch.no_grad():
        for batch_id, (input_ids, attn_masks, token_type_ids) in enumerate(data_loader):
            input_ids, attn_masks, token_type_ids = \
                input_ids.to(device), attn_masks.to(device), token_type_ids.to(device)

            logits = net(input_ids, attn_masks, token_type_ids)
            probs = torch.sigmoid(logits).squeeze(-1)
            preds = (probs >= threshold).int()
            res_preds.extend(preds)

    res_preds = torch.stack(res_preds).cpu().numpy()
    return res_preds
