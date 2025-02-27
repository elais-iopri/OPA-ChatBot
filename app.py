import streamlit as st
import datetime
from uuid import uuid4
from src.utils import stream_response
from src.streamlit import get_vector_index, get_open_router_llm, get_chat_opa
from typing import (
    Optional,
    List
)

# Get open router llm
llm = get_open_router_llm()

# create vector index
vector_index = get_vector_index()

# Create chat opa instance
chat_opa = get_chat_opa(
    _openai = llm,
    _vector_index = vector_index,
)

# Initialize chat histories
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = []

# Initialize chat memory
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []

# Initialize session id
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

if "previous_chat_id" not in st.session_state:
    st.session_state.previous_chat_id = None

# Logo OPA
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image("./assets/logo_opa.png", width=300)
    st.write("")  # Baris kosong pertama
    st.write("")  # Baris kosong kedua
    st.write("")  # Baris kosong pertama
    st.write("")  # Baris kosong kedua

greetings = "Halo, saya OPA, Pakar Sawit Anda dari PT. RPN, Pusat Penelitian Sawit. Apakah ada yang bisa saya bantu?"
st.chat_message(name="assistant", avatar= "./assets/OPA_avatar.jpeg").markdown(greetings)

# Displaying all historical messages
for num, message in enumerate(st.session_state.chat_histories):
    
    # message_role = list(message["chat_messages"].keys())
    
    with st.chat_message(name= "user", avatar= "./assets/user_avatar.jpeg") :
        st.markdown(st.session_state.chat_histories[num]["chat_messages"]["user"])

    with st.chat_message(name= "assistant", avatar= "./assets/OPA_avatar.jpeg") :
        st.markdown(st.session_state.chat_histories[num]["chat_messages"]["assistant"])

# Getting chat input from user
prompt = st.chat_input()

# Displaying chat prompt
if prompt:
    _chat_id = str(uuid4())

    # Displaying user chat prompt
    with st.chat_message(name="user", avatar="./assets/user_avatar.jpeg"):
        st.markdown(prompt)

    try :
        # Getting response from llm model
        response = chat_opa.get_response(
            question = prompt,
            chat_histories = st.session_state.chat_memory,
        )

        # Displaying response
        with st.chat_message("assistant", avatar="./assets/OPA_avatar.jpeg"):
            response = st.write_stream(stream_response(response))
            
            chat_history_data = {
                "chat_id" : _chat_id,
                "chat_messages" : {
                    "user" : prompt,
                    "assistant" : response
                },
                "conversation_id" : st.session_state.session_id,
                "created_at" : datetime.datetime.now(tz=datetime.timezone.utc),
                "previous_chat_id" : st.session_state.previous_chat_id
            }
                
            st.session_state.chat_histories.append(
                chat_history_data
            )

            index_latest_chat = len(st.session_state.chat_histories) - 1

            st.session_state.previous_chat_id = _chat_id

        # Just use 3 latest chat to chat history
        if len(st.session_state.chat_histories) > 3:
            st.session_state.chat_memory = [] = st.session_state.chat_histories[-3:]
        else:
            st.session_state.chat_memory = st.session_state.chat_histories
    except Exception as e:
        st.error(e)