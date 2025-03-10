import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from uuid import uuid4
from src.utils import stream_response, get_public_ip, preprocess_image
from src.streamlit import (
    get_vector_index,
    get_open_router_llm,
    get_chat_opa,
    get_cnn_model,
    send_feedback)
from src.database import (
    check_ip_already_exists,
    save_new_user,
    get_user_by_id,
    save_conversation,
    save_chat_history,
    save_thumb_chat_feedback,
    get_chat_histories,
    get_conversations
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

cnn_model = get_cnn_model()

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

st.session_state.user_conversations = get_conversations(st.session_state.user_session["id"])

if "exist_conversation" not in st.session_state:
    st.session_state.exist_conversation = None

# Seesion for save conversation logic
if "conversation_session_id" not in st.session_state:
    st.session_state.conversation_session_id = str(uuid4())

# Initialize chat histories
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = []

if st.session_state.exist_conversation is not None:
    # get chat history
    st.session_state.chat_histories = get_chat_histories(st.session_state.exist_conversation)

if "image_prediction" not in st.session_state:
    st.session_state.image_prediction = None

st.markdown(
    """
<style>
section.stSidebar > div {
    background-color: #01b6a2;
    background-image: none;
    color: #FFFFFF;
}
</style>
""",
    unsafe_allow_html=True,
)

# Sidebar For Navigating to previous conversation
with st.sidebar:    
    st.markdown(
        """
# Chat OPA
"""
    )
     
    if st.button("💬  **Percakapan Baru**", use_container_width=True, type="primary"):
        st.session_state.exist_conversation = None
        st.session_state.previous_chat_id = None
        st.session_state.conversation_saved = False
        st.session_state.conversation_session_id = str(uuid4())
        st.session_state.chat_histories = []
        st.rerun()

    """
    ---
    """

    st.write("## 💬 Percakapan Terdahulu")

    if len(st.session_state.user_conversations) > 0 :
        st.markdown(
    """
    <style>

    div.stButton > button {
        text-align: left !important;
        display: block;
    }

    </style>
    """,
    unsafe_allow_html=True
)
        for num, conversation in enumerate(st.session_state.user_conversations): # Bukan conversation histories melainkan session berbeda conversation!
            label = conversation["message_user"]
            if len(label) > 33:
                label = "**" + label[:31] + " . . ." + "**"
            else:
                label = "**" + label + "**"
            if st.button(label, key = f"conv_btn_{num}", use_container_width=True, type="primary"):
                st.session_state.conversation_data = conversation
                st.session_state.exist_conversation = conversation["session_id"]
                st.session_state.conversation_saved = False
                st.rerun()
        
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
        user_id = st.session_state.user_session["id"]
        session_id = st.session_state.session_id

        send_feedback(user_id, session_id)

# Logo OPA
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image("./assets/logo_opa.png", width=300)
    st.write("\n\n\n")

greetings = "Halo, saya OPA, Pakar Sawit Anda dari PT. RPN, Pusat Penelitian Sawit. Apakah ada yang bisa saya bantu?"
st.chat_message(name="assistant", avatar= "./assets/OPA_avatar.jpeg").markdown(greetings)

# Displaying all historical messages
for num, message in enumerate(st.session_state.chat_histories):
    st.session_state.previous_chat_id = message["chat_id"]
        
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
prompt = st.chat_input(
    accept_file=True,
    file_type=["jpg", "jpeg", "png"]
)

if prompt and prompt.files:
    # Open the uploaded image using PIL
    image = Image.open(prompt.files[0])

    # Preprocess the image for EfficientNetV2M
    processed_image = preprocess_image(image)
    
    predictions = cnn_model.predict(processed_image)

    decoded_predictions = tf.keras.applications.efficientnet_v2.decode_predictions(predictions, top=1)[0][0]

    st.session_state.image_prediction = decoded_predictions[1]
    st.write(st.session_state.image_prediction)

# Displaying chat prompt
if prompt and prompt.text:
    if not prompt.files :
        st.session_state.image_prediction = None
    
    st.write(st.session_state.image_prediction, "OK")
        
    _chat_id = str(uuid4()) # new session chat_session

    # Displaying user chat prompt
    with st.chat_message(name="user", avatar="./assets/user_avatar.jpeg"):
        st.markdown(prompt.text)

    try :
        # Getting response from llm model
        response = chat_opa.get_response(
            question = prompt.text,
            chat_histories = st.session_state.chat_memory,
            image_prediction = st.session_state.image_prediction
        )

        # Displaying response
        with st.chat_message("assistant", avatar="./assets/OPA_avatar.jpeg"):
            response = st.write_stream(stream_response(response))
            
            chat_history_data = {
                "chat_id" : _chat_id,
                "chat_messages" : {
                    "user" : prompt.text,
                    "assistant" : response
                },
                "previous_chat_id" : st.session_state.previous_chat_id
            }
                
            # save chat conversation
            if st.session_state.exist_conversation is None:
                
                st.session_state.conversation_data = save_conversation(
                    {
                        "conversation_session_id" : st.session_state.conversation_session_id,
                        "user_id" : st.session_state.user_session["id"]
                    }
                )

                st.session_state.exist_conversation = st.session_state.conversation_data["session_id"]

                        
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

        # Just use 3 latest chat to chat history
        if len(st.session_state.chat_histories) > 3:
            st.session_state.chat_memory = st.session_state.chat_histories[-3:]
        else:
            st.session_state.chat_memory = st.session_state.chat_histories
        
        st.rerun()

    except Exception as e:
        st.error(e)