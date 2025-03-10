import os
import streamlit as st
import time
import tensorflow as tf
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.document_compressors import JinaRerank
from typing import Optional
from src.pipeline import ChatOPA
from src.database import save_general_feedback

@st.cache_resource
def get_vector_index(collection_name : str = "OPA_Chatbot") -> QdrantVectorStore:
    # get embedding model
    _embeddings = JinaEmbeddings(model_name="jina-embeddings-v3")
    
    # create vector index client
    vector_index = QdrantVectorStore.from_existing_collection(
        embedding=_embeddings,
        collection_name=collection_name, # we can change the dataset collection here
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"],
        prefer_grpc=True,
        https=True
    )
    return vector_index

@st.cache_resource
def get_open_router_llm(model: str = "meta-llama/llama-3.3-70b-instruct") -> ChatOpenAI:
    return ChatOpenAI(
        model_name=model,
        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
        openai_api_base=st.secrets["OPENROUTER_BASE_URL"],
        # Pass the provider preference as an extra parameter
        extra_body={
            "provider": {
                "order": ["Nebius"], # "specify provider preference"
                "allow_fallbacks" : True, # "Allow changing other providers, if the main provider is not available"
                "sort" : "price" # "Sort the provider based on price"
            }
        }
    )

@st.cache_resource
def get_chat_opa(_openai: ChatOpenAI, _vector_index : QdrantVectorStore, reranker : Optional[JinaRerank] = None ) -> ChatOPA:
    return ChatOPA(
        openai = _openai,
        vector_index = _vector_index,
        reranker = reranker
    )

@st.cache_resource
def get_cnn_model():
    model = tf.keras.applications.EfficientNetV2M(weights = "efficientnetv2-m.h5", input_shape=[224,224,3])
    return model
    
@st.dialog("Berikan Feedback")
def send_feedback(user_id : str, session_id: str):
    with st.form(key="feedback_input", enter_to_submit=False, clear_on_submit=False):
        name = st.text_input("Nama")
        bagian = st.text_input("Bagian")
        sub_bagian = st.text_input("Sub-Bagian")
        puslit = st.text_input("Puslit")
        feedback = st.text_area("Feedback")

        rating = [1, 2, 3, 4, 5]
        selected_rating = st.feedback(options="stars")

        if st.form_submit_button("Submit"):
            # Save data to Google Sheets
            if selected_rating is not None:
                data = {
                    "user_id" : user_id,
                    "session" : session_id,
                    "name" : name,
                    "bagian" : bagian,
                    "sub_bagian" : sub_bagian,
                    "puslit" : puslit,
                    "rating" : rating[selected_rating],
                    "feedback" : feedback,
                }

                save_general_feedback(data)

                st.success("Terimakasih atas umpan balik anda!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Tolong berikan rating 🙏")
