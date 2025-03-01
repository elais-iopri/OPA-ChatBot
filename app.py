import streamlit as st
from uuid import uuid4
from src.utils import stream_response, get_public_ip
from src.streamlit import (
    get_vector_index, get_open_router_llm, get_chat_opa,
    send_feedback)

from src.database import (
    check_ip_already_exists,
    save_new_user,
    get_user_by_id,
    save_conversation,
    save_chat_history,
    save_thumb_chat_feedback
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

# Seesion for save conversation logic
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid4())

if "conversation_saved" not in st.session_state:
    st.session_state.conversation_saved = False

if "conversation_data" not in st.session_state:
    st.session_state.conversation_data = None

# Initialize chat memory
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []

# Initialize session id
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

if "previous_chat_id" not in st.session_state:
    st.session_state.previous_chat_id = None

# Store IP Client
if "ip_address" not in st.session_state:
    st.session_state.ip_address = get_public_ip()

# Saving user session
if "user_session" not in st.session_state:
    st.session_state.user_session = check_ip_already_exists(st.session_state.ip_address)

if "_temp_feedback" not in st.session_state:
    st.session_state._temp_feedback = None

# Check if IP already exists
if st.session_state.user_session is None:
    _user_id = save_new_user(st.session_state.ip_address, st.session_state.session_id)
    st.session_state.user_session = get_user_by_id(_user_id)
    
# Feedback Form
with st.expander("Chat OPA", icon=":material/priority_high:", expanded=False):
    st.markdown(body=
"""
Chat OPA adalah asisten virtual yang akan membantu anda terkait kultur kelapa sawit.

**Aplikasi** ini sedang dalam pengembangan dan memerlukan **Feedback** dari pengguna.

Silahkan coba untuk menanyakan sesuatu seputar kultur kelapa sawit. Setelah itu, mohon untuk mengisi *Feedback Form* dibawah ini.
"""
)

    if st.button("Feedback Form", type="primary"):
        send_feedback()

# Logo OPA
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image("./assets/logo_opa.png", width=300)
    st.write("\n\n\n")

greetings = "Halo, saya OPA, Pakar Sawit Anda dari PT. RPN, Pusat Penelitian Sawit. Apakah ada yang bisa saya bantu?"
st.chat_message(name="assistant", avatar= "./assets/OPA_avatar.jpeg").markdown(greetings)

# Displaying all historical messages
for num, message in enumerate(st.session_state.chat_histories):
        
    with st.chat_message(name= "user", avatar= "./assets/user_avatar.jpeg") :
        st.markdown(message["message_user"])

    with st.chat_message(name= "assistant", avatar= "./assets/OPA_avatar.jpeg") :
        st.markdown(message["message_assistant"])

    
        if message["thumb_score"] is None:
            selected_thumb_feedback = st.feedback (
                options="thumbs",
                key=f"fb_{num}",
                on_change=save_thumb_chat_feedback,
                args=[num, message["id"]]
                )

        else :
            if st.session_state.chat_histories[num]["thumb_score"] == 1:
               st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Material+Icons" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Icons+Outlined" rel="stylesheet">

<i class="material-icons" style="font-size:20px; color:#01b6a2;">thumb_up</i>
<i class="material-icons-outlined" style="font-size:20px; color:#31333f99;">thumb_down</i>
                           
 """, unsafe_allow_html=True)
            
            else :
                st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Material+Icons" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Icons+Outlined" rel="stylesheet">

<i class="material-icons-outlined" style="font-size:20px; color:#31333f99;">thumb_up</i>
<i class="material-icons" style="font-size:20px; color:#fa6e00;">thumb_down</i>
""", unsafe_allow_html=True)

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
                "previous_chat_id" : st.session_state.previous_chat_id
            }
                
            # save chat conversation
            if not st.session_state.conversation_saved:
                st.session_state.conversation_data = save_conversation(
                    {
                        "conversation_id" : st.session_state.conversation_id,
                        "user_id" : st.session_state.user_session["id"]
                    }
                )
                
                st.session_state.conversation_saved = True

            # save chat history
            _chat_temp = save_chat_history(chat_history_data, st.session_state.conversation_data)

            if _chat_temp is not None:
                st.session_state.chat_histories.append(_chat_temp)
                del _chat_temp
            
            index_latest_chat = len(st.session_state.chat_histories) - 1
            
            if st.session_state.chat_histories[index_latest_chat]["thumb_score"] is None:
                selected_thumb_feedback = st.feedback (
                    options="thumbs",
                    key=f"fb_{index_latest_chat}",
                    on_change=save_thumb_chat_feedback,
                    args=[index_latest_chat, st.session_state.chat_histories[index_latest_chat]["id"]]
                    )
                
            else :
                if st.session_state.chat_histories[index_latest_chat]["thumb_score"] == 1:
                    st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Material+Icons" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Icons+Outlined" rel="stylesheet">

<i class="material-icons" style="font-size:20px; color:green;">thumb_up</i>
<i class="material-icons-outlined" style="font-size:20px; color:black;">thumb_down</i>
                           
 """, unsafe_allow_html=True)
            
                else :
                    st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Material+Icons" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Icons+Outlined" rel="stylesheet">

<i class="material-icons-outlined" style="font-size:20px; color:black;">thumb_up</i>
<i class="material-icons" style="font-size:20px; color:red;">thumb_down</i>
""", unsafe_allow_html=True)
    
            st.session_state.previous_chat_id = _chat_id

        # Just use 3 latest chat to chat history
        if len(st.session_state.chat_histories) > 3:
            st.session_state.chat_memory = [] = st.session_state.chat_histories[-3:]
        else:
            st.session_state.chat_memory = st.session_state.chat_histories

    except Exception as e:
        st.error(e)