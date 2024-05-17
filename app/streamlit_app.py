"""Streamlit-приложение для поиска похожих вопросов."""

import streamlit as st
st.set_page_config(layout='wide')
import requests
import json
import pandas as pd


st.title('Similar Medical Questions Search')
st.header('Enter your text in English and how many similar medical questions you want to find:')

user_text = st.text_area("Your question (100 characters limit): ", max_chars=100, key="input")
questions_num = st.number_input('Number of similar questions to find (20 questions limit):', 
                                min_value=1, 
                                max_value=20, 
                                value=10, 
                                step=1)
is_cls_needed = st.checkbox("Whether to use a pretrained classifier to filter search results")

if st.button('Search') and user_text:
    query_params = {
        'user_text': user_text,
        'questions_num': questions_num,
        'is_cls_needed': is_cls_needed
    }
    state = st.text('Search in process...')
    response = requests.post(url="http://127.0.0.1:8081/similar_questions/", params=query_params)
    response = response.json()
    st.table(pd.DataFrame(response['similar_questions'], columns=['Similar Medical Questions']))
    st.text(f"Total run time: {round(response['time'], 2)}")
    state.text('Search is done.')
