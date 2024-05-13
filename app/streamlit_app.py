"""Streamlit-приложение для поиска похожих вопросов."""

import streamlit as st
st.set_page_config(layout='wide')
import requests
import json


st.title('Similar Medical Questions Search')
st.header('Enter your medical question and a number of similar ones you want to find:')

user_text = st.text_area("Your question (100 characters limit): ", max_chars=100, key="input")
questions_num = st.number_input('Number of similar questions (20 questions limit):', 
                                min_value=1, 
                                max_value=20, 
                                value=10, 
                                step=1)

if st.button('Search similar questions') and user_text:
    query_params = {
        'user_text': user_text,
        'questions_num': questions_num
    }
    state = st.text('Search in process...')
    response = requests.post(url='http://127.0.0.1:8081/similar_questions', data=json.dumps(query_params))
    state.text('Search is done.')
    st.write(response['similar_questions'])
