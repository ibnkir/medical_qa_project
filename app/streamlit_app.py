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

user_text = st.text_area("Your question (150 characters limit): ", 
                         value="How to cope with stress?", 
                         max_chars=150, 
                         key="input")

questions_num = st.number_input('Number of similar medical questions to find (20 questions limit):',
                                min_value=1, 
                                max_value=20, 
                                value=10, 
                                step=1)

cls_option = st.selectbox(
    "Select your classifier (without GPU BERT takes too much time):",
    ("None", "CatBoostClassifier", "Fine-tuned BERT"))

if st.button('Search') and user_text:
    with st.spinner('Search in process...'):
        query_params = {
            'user_text': user_text,
            'questions_num': questions_num,
            'cls_option': cls_option
        }
    
        #state = st.text('Search in process...')
    
        try:
            response = requests.post(url="http://127.0.0.1:8081/similar_questions/", params=query_params)
            response = response.json()
            
            if response['status'] == 'OK':
                st.table(pd.DataFrame(response['similar_questions'], columns=['Similar Medical Questions']))
                st.text(f"Search is done. {response['message']}")
                st.text(f"Total run time: {round(response['time'], 2)}")
            else:
                st.text(f"Error: {response['message']}")

        except Exception as e:
            st.text(f'Error: {e}')
