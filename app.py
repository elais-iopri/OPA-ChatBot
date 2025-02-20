# Installing Dependencies
import os
import json
import streamlit as st
import gspread
import time
import pytz
import random
from uuid import uuid4
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from typing import Tuple, List
from pydantic import BaseModel, Field, field_validator, ConfigDict
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_groq import ChatGroq
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
st.session_state.session_id = str(uuid4()) 

if 'messages_product_knowledge' not in st.session_state:
    st.session_state.messages_product_knowledge = []

if '_log' not in st.session_state:
    st.session_state['_log'] = []

if 'chat_history_product_knowledge' not in st.session_state:
    st.session_state.chat_history_product_knowledge = []

if 'need_greetings_product_knowledge' not in st.session_state:
    st.session_state.need_greetings_product_knowledge = True

if 'convert_status' not in st.session_state:
    st.session_state.convert_status = None

if 'conversion_done' not in st.session_state:
    st.session_state.conversion_done = None

if 'conversion_running' not in st.session_state:
    st.session_state.conversion_running = None

if 'idx_llm' not in st.session_state:
    st.session_state['idx_llm'] = 0

if 'total_time' not in st.session_state:
    st.session_state['total_time'] = 0

# st.write(st.session_state.convert_status)
if st.session_state.conversion_done is not None:
    if st.session_state.conversion_done:
        st.toast("Document conversion finished!", icon="✅")  # Or use st.success
        st.session_state.conversion_done = False  # Reset to avoid repeated toasts

if 'current_main_key_idx' not in st.session_state:
    st.session_state.current_main_key_idx = random.randint(0, 4)  # Pilih main key secara acak

if 'use_backup' not in st.session_state:
    st.session_state.use_backup = False

if 'backup_idx' not in st.session_state:
    st.session_state.backup_idx = -1

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
    sheet.append_row([datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S"), sesssion_id, name, bidang, rating, feedback, conversation])


# Load llm model using Groq
@st.cache_resource
def load_llm_groq(KEY):
    return ChatGroq(
        model='llama-3.3-70b-versatile', #llama-3.1-70b-versatile, llama-3.1-8b-instant
        temperature=0,
        api_key=KEY
    )

llm_groq = load_llm_groq(st.secrets['GROQ_API_KEY'])

# Integrate with Vector DB
@st.cache_resource
def create_vector_space():
    embeddings = JinaEmbeddings(model_name="jina-embeddings-v3")
    vector_index = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="OPA_Chatbot",
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"],
        prefer_grpc=True,
        https=True
    )
    return vector_index

vector_index = create_vector_space()

# # Instantiate jina reranker model
# @st.cache_resource
# def load_jina_reranker():
#     return JinaRerank(model_name="jina-reranker-v2-base-multilingual")

# jina_reranker = load_jina_reranker()

def retrieve_context_by_vector(question):
    question = remove_lucene_chars_cust(question)
    return [el for el in vector_index.similarity_search(question, k=4)]

def retrieve_context_by_vector_with_reranker(question):
    question = remove_lucene_chars_cust(question)

    retriever = vector_index.as_retriever(
        search_type = 'similarity',
        search_kwargs = {
            'k' : 20,
        }
    )
    
    compressor = JinaRerank(model="jina-reranker-v2-base-multilingual")

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    compressed_docs = compression_retriever.invoke(
        question
    )

    return compressed_docs



# Retrival knowledge
def retriever(question: str):
    unstructured_data = retrieve_context_by_vector_with_reranker(question)

    documents = []
    
    for doc in unstructured_data:
        sections = ""
        
        if "Header 1" in doc.metadata:
            sections += f"Header 1 - {doc.metadata['Header 1']}\n"

        if "Header 2" in doc.metadata:
            sections += f"Header 2 - {doc.metadata['Header 2']}\n"

        if "Header 3" in doc.metadata:
            sections += f"Header 3 - {doc.metadata['Header 3']}\n"

        if "Header 4" in doc.metadata:
            sections += f"Header 4 - {doc.metadata['Header 4']}\n"

        if "Header 5" in doc.metadata:
            sections += f"Header 5 - {doc.metadata['Header 5']}\n"

        documents.append(
            f"""
Section :
{sections}
Content :
{doc.page_content.replace("text: ", "")}
"""
        )
    nl = "\n---\n"
    final_data = f"""

Unstructured data:
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
        | llm_groq
        | StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(lambda x : x["question"]),
)

# Prompt to real prompt
template = """Your name is OPA. You are a great, friendly and professional AI chat bot about product from the "Central of Oil Palm Research".

### User Question:
{question}

### Retrieved Context:
{context}

### Important Instructions:
- Base your response only on the provided context. If the contexts provided do not match, say you don't know.

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
    | llm_groq
    | StrOutputParser()
)

def stream_response(response, delay=0.02):
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
                save_feedback_to_google_sheets(st.session_state.session_id, name, bidang, rating[selected_rating], feedback, st.session_state.messages_product_knowledge)
                st.success("Terimakasih atas umpan balik anda!")
            else:
                st.error("Tolong berikan rating 🙏")
            

with st.expander("OPA - Pakar Sawit", icon=":material/priority_high:", expanded=True):
    st.markdown(body=
"""
PAKAR SAWIT adalah asisten virtual yang akan membantu anda terkait kultur kelapa sawit.

**Aplikasi** ini sedang dalam pengembangan dan memerlukan **Feedback** dari pengguna.

Silahkan coba untuk menanyakan sesuatu seputar kultur kelapa sawit. Setelah itu, mohon untuk mengisi *Feedback Form* dibawah ini.
"""
)

    if st.button("Feedback Form", type="primary"):
        send_feedback()

st.image(image="./assets/Logo-RPN.png", width=240)
st.header("(OPA) - Pakar Sawit", divider="gray")

# Displaying all historical messages
for message in st.session_state.messages_product_knowledge:
    st.chat_message(message['role']).markdown(message['content'])

if st.session_state.need_greetings_product_knowledge :
    # greet users
    greetings = "Selamat Datang, Saya adalah OPA, asisten virtual yang akan membantu anda terkait kultur kelapa sawit. Apakah ada yang bisa saya bantu?"
    st.chat_message("assistant",).markdown(greetings)

    st.session_state.messages_product_knowledge.append({'role' : 'assistant', 'content': greetings})

    st.session_state.need_greetings_product_knowledge = False


# Getting chat input from user
prompt = st.chat_input()


# Displaying chat prompt
if prompt:
    # Displaying user chat prompt
    with st.chat_message("user"):
        st.markdown(prompt)

    try :
        # Getting response from llm model
        response = chain.stream({
            "chat_history" : st.session_state.chat_history_product_knowledge, 
            "question" : prompt
        })
        
        # Saving user prompt to session state
        st.session_state.messages_product_knowledge.append({'role' : 'user', 'content': prompt})

        # Displaying response
        with st.chat_message("assistant",):
            response = st.write_stream(stream_response(response))

        # Saving response to chat history in session state
        st.session_state.messages_product_knowledge.append({'role' : 'assistant', 'content': response})

        # Saving user and llm response to chat history
        st.session_state.chat_history_product_knowledge.append((prompt, response))

        # Just use 3 latest chat to chat history
        if len(st.session_state.chat_history_product_knowledge) > 3:
            st.session_state.chat_history_product_knowledge = st.session_state.chat_history_product_knowledge[-3:]

    except Exception as e:
        st.error(e)   
