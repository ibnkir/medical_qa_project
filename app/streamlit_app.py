"""Streamlit-приложение для поиска похожих вопросов.

Для запуска перейти в папку app и выполнить команду
streamlit run streamlit_app.py
"""

import streamlit as st
st.set_page_config(layout='wide')
import requests
import json
import pandas as pd


st.title('Similar Medical Questions Search')
st.write(f"### {'Enter your question in English and how many similar medical questions to find:'}")

user_text = st.text_area("Your question (100 characters limit): ", 
                         value="How do I check my blood sugar?", 
                         max_chars=100, 
                         key="input")

questions_num = st.number_input('Number of similar questions to find (20 questions limit):',
                                min_value=1, 
                                max_value=20, 
                                value=10, 
                                step=1)

cls_option = st.selectbox(
    "Select your classifier:",
    ("None", "CatBoostClassifier", "Fine-tuned BERT"))

if st.button('Search') and user_text:
    query_params = {
        'user_text': user_text,
        'questions_num': questions_num,
        'cls_option': cls_option
    }
    
    state = st.text('Search in process...')
    try:
        response = requests.post(url="http://127.0.0.1:8081/similar_questions/", params=query_params)
        response = response.json()
        
        if response['status'] == 'OK':
            st.table(pd.DataFrame(response['similar_questions'], columns=['Similar Medical Questions']))
            st.text(f"Total run time: {round(response['time'], 2)}")
            state.text(f"Search is done. {response['message']}")
        else:
            state.text(f"Error: {response['message']}")

    except Exception as e:
        state.text(f'Error: {e}')
