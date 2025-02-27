import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.document_compressors import JinaRerank
from typing import Optional
from src.pipeline import ChatOPA

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



