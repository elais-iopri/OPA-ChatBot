# Installing Dependencies
import os
import json
import streamlit as st
import gspread
import time
import pytz
import random
import requests
import datetime
import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials
from uuid import uuid4
from oauth2client.service_account import ServiceAccountCredentials
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from google.cloud.firestore_v1.types.write import WriteResult
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from typing import Tuple, List, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from os import getenv
from dotenv import load_dotenv
from langchain_community.embeddings import JinaEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import JinaRerank
from langchain_qdrant import QdrantVectorStore


st.set_page_config(
    page_title="OPA Chat",
    layout="centered",
)


# Load environtment app
load_dotenv()

# Start Counter
start_counter = time.perf_counter()

# Setup a session state to hold up all the old messages
# Setting up session id
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid4())

if 'conversation_saved' not in st.session_state:
    st.session_state.conversation_saved = False

if 'messages' not in st.session_state:
    st.session_state.messages = []

if '_log' not in st.session_state:
    st.session_state['_log'] = []

if 'chat_histories' not in st.session_state:
    st.session_state.chat_histories = []

if 'chat_histories_to_save' not in st.session_state:
    st.session_state.chat_histories_to_save = []

if 'need_greetings' not in st.session_state:
    st.session_state.need_greetings = True

if 'previous_chat_id' not in st.session_state:
    st.session_state.previous_chat_id = None

def remove_lucene_chars_cust(text: str) -> str:
    """Remove Lucene special characters"""
    special_chars = [
        "+",
        "-",
        "&",
        "|",
        "!",
        "(",
        ")",
        "{",
        "}",
        "[",
        "]",
        "^",
        '"',
        "~",
        "*",
        "?",
        ":",
        "\\",
        "/"
    ]

    for char in special_chars:
        if char in text:
            # if char == "/":
            #     text = text.replace(char, "\\/")
            # else :
            text = text.replace(char, " ")
    
    return text.strip()

# Function to get IP address
def get_client_ip():
    try:
        response = requests.get("https://api64.ipify.org?format=json")
        return response.json()["ip"]
    except Exception as e:
        return f"Error: {e}"
    

if 'ip_client' not in st.session_state:
    st.session_state.ip_client = get_client_ip()
    
@st.cache_resource
def connect_to_firebase_client():
    # Use a service account
    cred = credentials.Certificate(st.secrets["firebase_key"].to_dict())
    firebase_admin.initialize_app(cred)

    return firestore.client()

db = connect_to_firebase_client()

@st.cache_resource
def connect_to_google_sheets():
    # Define the scope
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    # Authenticate credentials
    credentials_dict = st.secrets["gspread_credential"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(credentials_dict, scope)
    client = gspread.authorize(creds)
    return client

# Save feedback to Google Sheets
def save_feedback_to_google_sheets(name,sesssion_id, bidang, rating, feedback, chat_message):
    # Connect to Google Sheets
    client = connect_to_google_sheets()
    sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/17nV0tRr0sQLJlTD3BopuZFybSjL6R2C8kCCNYYzxdyg/edit?usp=sharing").sheet1# Open the Google Sheet by name
    
    chats = []
    separator = "\n---\n"

    if len(chat_message) <= 1:
        conversation = rf""
    else:
        for chat in chat_message[1:]:
            role = chat["role"]
            content = chat["content"]
            chats.append(
                f"{role}:{content}"
            )

        conversation = f"""
{separator.join([_chat for _chat in chats])}
"""
    # print(conversation)
    # Append the feedback
    sheet.append_row([datetime.datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S"), sesssion_id, name, bidang, rating, feedback, conversation])


@st.cache_resource
def get_openrouter_llm(model: str = "meta-llama/llama-3.3-70b-instruct") -> ChatOpenAI:
    return ChatOpenAI(
        model_name=model,
        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
        openai_api_base=st.secrets["OPENROUTER_BASE_URL"],
        # Pass the provider preference as an extra parameter
        extra_body={
            "provider": {
                "order": ["Groq"], # "specify provider preference"
                "allow_fallbacks" : True, # "Allow changing other providers, if the main provider is not available"
                "sort" : "price" # "Sort the provider based on price"
            }
        }
    )

llm = get_openrouter_llm()

# Integrate with Vector DB
@st.cache_resource
def create_vector_space():
    embeddings = JinaEmbeddings(model_name="jina-embeddings-v3")
    vector_index = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="KSI_ProductKnowledge23",
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"],
        prefer_grpc=True,
        https=True
    )
    return vector_index

vector_index = create_vector_space()

def retrieve_context_by_vector(question):
    question = remove_lucene_chars_cust(question)
    return [el for el in vector_index.similarity_search(question, k=4)]

def retrieve_context_by_vector_with_score_and_rerank(question: str, 
                                                     k_initial: int = 10, 
                                                     k_final: int = 3, 
                                                     relevance_threshold: float = 0.5):
    """
    Mengambil konteks dari vector store dengan dua tahap:
    1. Mengambil hasil awal beserta skor relevansi menggunakan similarity_search_with_score.
    2. Menyaring dokumen berdasarkan ambang skor, lalu melakukan reranking menggunakan JinaRerank.
    
    Parameter:
    - question: Pertanyaan pengguna.
    - k_initial: Jumlah dokumen awal yang diambil.
    - k_final: Jumlah dokumen akhir yang dikembalikan setelah reranking.
    - relevance_threshold: Ambang skor untuk menyaring dokumen.
    
    Fungsi ini mencetak nilai skor dari hasil awal, kemudian mencetak konten hasil reranking.
    """
    # Bersihkan pertanyaan
    cleaned_question = remove_lucene_chars_cust(question)
    
    # Ambil hasil awal dengan skor relevansi
    initial_results = vector_index.similarity_search_with_score(cleaned_question, k=k_initial)
    
    # print("=== Hasil retrieval awal ===")
    # for idx, (doc, score) in enumerate(initial_results, start=1):
    #     print(f"Dokumen {idx}: Score = {score}")
    #     print(f"Content: {doc.page_content}\n")
    
    # Saring dokumen berdasarkan ambang skor
    filtered_docs = [doc for (doc, score) in initial_results if score >= relevance_threshold]
    
    # Bungkus fungsi penyedia dokumen dalam RunnableLambda (sesuai ekspektasi ContextualCompressionRetriever)
    base_retriever_runnable = RunnableLambda(lambda q: filtered_docs)
    
    # Buat reranker dengan model JinaRerank
    compressor = JinaRerank(model="jina-reranker-v2-base-multilingual")
    
    # Buat pipeline retriever dengan reranking
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever_runnable
    )
    
    # Lakukan reranking
    reranked_docs = compression_retriever.invoke(cleaned_question)
    
    # Ambil hanya k_final dokumen teratas
    final_docs = reranked_docs[:k_final]
    
    # print("=== Hasil reranking ===")
    # for idx, doc in enumerate(final_docs, start=1):
    #     print(f"Reranked Dokumen {idx}:")
    #     print(f"{doc.page_content}\n")
    
    return final_docs

# Retrival knowledge
def retriever(question: str):
    unstructured_data = retrieve_context_by_vector(question)

    documents = []

    # Not Include section header because the page content already include the section header
    
    for doc in unstructured_data:
        # sections = ""
        
        # if "Header 1" in doc.metadata:
        #     sections += f"Header 1 - {doc.metadata['Header 1']}\n"

        # if "Header 2" in doc.metadata:
        #     sections += f"Header 2 - {doc.metadata['Header 2']}\n"

        # if "Header 3" in doc.metadata:
        #     sections += f"Header 3 - {doc.metadata['Header 3']}\n"

        # if "Header 4" in doc.metadata:
        #     sections += f"Header 4 - {doc.metadata['Header 4']}\n"

        # if "Header 5" in doc.metadata:
        #     sections += f"Header 5 - {doc.metadata['Header 5']}\n"

        # Section :
        # {sections}
        documents.append(
            f"""
Content :
{doc.page_content.replace("text: ", "")}
    """
        )
    nl = "\n---\n"
    final_data = f"""

{nl.join(documents)}

"""

    print(final_data)
    return final_data

# Reference:
# {new_line.join(references)}
_template = """
You are an assistant skilled in paraphrasing questions, ensuring they align with the current conversation context. Every time a new question appears, check the recent chat history to decide if it's on the same topic or if there's a new topic shift. 

Guidelines:
1. If the latest question is vague (e.g., "What is its capital?"), identify the most recent *explicitly mentioned topic* in the chat history and use it as context.
2. When a new complete question introduces a different topic, assume it's a topic shift and use this new topic in the next responses until another shift occurs.
3. Prioritize the most recent complete topic if multiple topics are discussed in history.

**Examples:**

Example 1:
**Chat History:**
- User: "Who is the president of Indonesia?"
- AI: "The president of Indonesia is Joko Widodo."

**Latest Question:**  
User: "When did it gain independence?"

**Paraphrased Question:**  
"When did Indonesia gain independence?"

---

Example 2 (Topic Shift):
**Chat History:**
- User: "Who is the president of Indonesia?"
- AI: "The president of Indonesia is Joko Widodo."
- User: "What is its capital?"
- AI: "The capital of Indonesia is Jakarta."
- User: "Who is the president of Vietnam?"
- AI: "The president of Vietnam is Tran Dai Quang."

**Latest Question:**  
User: "What is its capital?"

**Paraphrased Question:**  
"What is the capital of Vietnam?"

---

Example 3:
**Chat History:**
- User: "Who is the CEO of Apple?"
- AI: "The CEO of Apple is Tim Cook."
  
**Latest Question:**  
User: "How many employees does it have?"

**Paraphrased Question:**  
"How many employees does Apple have?"

---

Example 4 (Topic Shift):
**Chat History:**
- User: "Who is the CEO of Apple?"
- AI: "The CEO of Apple is Tim Cook."
- User: "What is the companys revenue?"
- AI: "Apple's revenue is $274.5 billion."

**Latest Question:**  
User: "What is his revenue?"

**Paraphrased Question:**  
"What is the revenue of CEO Apple?"

---

Now, parafrase the latest question based on the recent topic or topic shift, using the latest chat history provided.
But don't explain in  output. just give the parafrased question as output.

**Chat History:**
{chat_history}

**Latest Question:**
{question}

**Paraphrased Question:**
"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# Chat history fromatter
def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

# Extract chat history if exists
_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),  # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(lambda x : x["question"]),
)

# Prompt to real prompt
template = """Your name is OPA. You are a great, friendly and professional AI chat bot about product from the "Central of Oil Palm Research".

### User Question:
{question}

### Context:
{context}

### Important Instructions:
- Base your response only on the provided context. If the contexts provided do not match or don't exist, say you don't know.
- When answering questions, do not include a greeting or introduction unless explicitly requested.

Your Answer: """

prompt = ChatPromptTemplate.from_template(template)


# Creating chain for llm
chain = (
    RunnableParallel(
        {
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)

def stream_response(response, delay=0.01):
    for res in response:
        yield res
        time.sleep(delay)


@st.dialog("Berikan Feedback")
def send_feedback():
    with st.form(key="feedback_input", enter_to_submit=False, clear_on_submit=False):
        name = st.text_input("Nama")
        bidang = st.text_input("Bidang")
        feedback = st.text_area("Feedback")

        rating = [1, 2, 3, 4, 5]
        selected_rating = st.feedback(options="stars")

        # print("INI FEEDBACK: ", feedback)
        if st.form_submit_button("Submit"):
            # Save data to Google Sheets
            if selected_rating is not None:
                # sesssion_id, name, bidang, rating, feedback, conversation
                save_feedback_to_google_sheets(st.session_state.session_id, name, bidang, rating[selected_rating], feedback, st.session_state.messages)
                st.success("Terimakasih atas umpan balik anda!")
            else:
                st.error("Tolong berikan rating 🙏")
            

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

def dialog_on_change(chat_id, index):
    st.session_state.chat_histories_to_save[index]["feedback"] = st.session_state[f"text_area_{index}"]

def save_text_feedback(collection_id, chat_id, feedback):
    conversation_ref = db.collection("conversations").document(collection_id).collection("chat_histories").document(chat_id)

    conversation_ref.update({
        "feedback" : feedback
    })

@st.dialog("Feedback")
def give_feedback_chat(chat_id, index):
    st.write(f"Anda memberikan 👎 dari respon chatbot")
    # st.write(chat_id)

    feedback = st.text_area(
            label="Beritahu kami",
            placeholder="Tuliskan alasan kenapa memberikan feedback 👎",
            key=f"text_area_{index}",
            on_change=dialog_on_change,
            args = [chat_id, index]
        )
    
    if st.button("Submit"):
        if "feedback" in st.session_state.chat_histories_to_save[index] :
            if len(st.session_state.chat_histories_to_save[index]["feedback"].strip()) > 0 :
                save_text_feedback(st.session_state.session_id, chat_id, feedback)
                st.success("Terimakasih atas feedback anda", icon="👍")
                time.sleep(1)
                st.rerun()
            else :
                st.error("Silahkan isi feedback terlebih dahulu", icon="🚫")
        else :
            st.error("Silahkan isi feedback terlebih dahulu", icon="🚫")


def save_chat_feedback(index, chat_id, collection_id):
    thumb_mapping = ["DOWN", "UP"]

    thumb_score = thumb_mapping[st.session_state[f"fb_{index}"]]
    st.session_state.chat_histories_to_save[index]["thumb_score"] = thumb_score # thumb score saved in session state

    # update database
    conversation_ref = db.collection("conversations").document(collection_id).collection("chat_histories").document(chat_id)

    # update field
    conversation_ref.update({
        "thumb_score" : thumb_score
    })

    if thumb_mapping[st.session_state[f"fb_{index}"]] == "DOWN" :
        give_feedback_chat(chat_id, index)

def save_chat_collection(collection_id : str, ip_client : str) -> WriteResult:
    """
        collection_id => conversation_id (got from session id)
    """
    data = {
        "conversation_id" : collection_id,
        "created_at" : datetime.datetime.now(tz=datetime.timezone.utc),
        "ip_client" : ip_client
    }

    conversation_ref = db.collection("conversations").document(collection_id) # create document with id => session id (UUID4)
    
    return conversation_ref.set(data) # add data to document
   

def save_chat_history(collection_id : str, chat_history : dict) -> WriteResult:
    """
        collectio_id => conversation_id,
        chat_history => dict of sub collection to save
    """

    conversation_ref = db.collection("conversations").document(collection_id) # Get conversation collection ref

    # Create new sub collection and new document
    return conversation_ref.collection("chat_histories").document(chat_history["chat_id"]).set(chat_history) # add data to document

# Logo OPA
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image("./assets/logo_opa.png", width=300)
    st.write("")  # Baris kosong pertama
    st.write("")  # Baris kosong kedua
    st.write("")  # Baris kosong pertama
    st.write("")  # Baris kosong kedua

# # st.image(image="./assets/Logo-RPN.png", width=240)
# st.header("(OPA) - Pakar Sawit", divider="gray")

greetings = "Halo, saya OPA, Pakar Sawit Anda dari PT. RPN, Pusat Penelitian Sawit. Apakah ada yang bisa saya bantu?"
st.chat_message(name="assistant", avatar= "./assets/OPA_avatar.jpeg").markdown(greetings)

# Displaying all historical messages
for num, message in enumerate(st.session_state.chat_histories_to_save):
    
    # message_role = list(message["chat_messages"].keys())
    
    with st.chat_message(name= "user", avatar= "./assets/user_avatar.jpeg") :
        st.markdown(st.session_state.chat_histories_to_save[num]["chat_messages"]["user"])

    with st.chat_message(name= "assistant", avatar= "./assets/OPA_avatar.jpeg") :
        st.markdown(st.session_state.chat_histories_to_save[num]["chat_messages"]["assistant"])

        if "thumb_score" not in message:
            selected_thumb_feedback = st.feedback (
                options="thumbs",
                key=f"fb_{num}",
                on_change=save_chat_feedback,
                args=[num, message["chat_id"], st.session_state.session_id]
                )

        else :
            if st.session_state.chat_histories_to_save[num]["thumb_score"] == "UP":
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
        response = chain.stream({
            "chat_history" : st.session_state.chat_histories, 
            "question" : prompt
        })

        # Displaying response
        with st.chat_message("assistant", avatar="./assets/OPA_avatar.jpeg"):
            response = st.write_stream(stream_response(response))

            # Saving user prompt to session state
            st.session_state.messages.append({'role' : 'user', 'content': prompt})

            # Saving response to chat history in session state
            st.session_state.messages.append({'role' : 'assistant', 'content': response})

            # Saving user and llm response to chat history
            st.session_state.chat_histories.append((prompt, response))

            
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
                
            st.session_state.chat_histories_to_save.append(
                chat_history_data
            )

            index_latest_chat = len(st.session_state.chat_histories_to_save) - 1
            
            if "thumb_score" not in st.session_state.chat_histories_to_save[index_latest_chat]:
                selected_thumb_feedback = st.feedback (
                    options="thumbs",
                    key=f"fb_{index_latest_chat}",
                    on_change=save_chat_feedback,
                    args=[index_latest_chat, st.session_state.chat_histories_to_save[index_latest_chat]["chat_id"], st.session_state.session_id]
                    )
                
            else :
                if st.session_state.chat_histories_to_save[num]["thumb_score"] == "UP":
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

            # save chat conversation
            if st.session_state.conversation_saved == False :
                save_chat_collection(st.session_state.session_id, st.session_state.ip_client)
                st.session_state.conversation_saved = True
            
            # save chat history
            save_chat_history(st.session_state.session_id, chat_history_data)

            st.session_state.previous_chat_id = _chat_id

        # Just use 3 latest chat to chat history
        if len(st.session_state.chat_histories) > 3:
            st.session_state.chat_histories = st.session_state.chat_histories[-3:]

    except Exception as e:
        st.error(e)   
