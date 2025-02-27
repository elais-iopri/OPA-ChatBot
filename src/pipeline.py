from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import JinaEmbeddings
from src.utils import remove_lucene_chars_cust
from langchain_community.document_compressors import JinaRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from typing import (
    List,
    Optional,
    Dict
)

# Define the pipeline
class ChatOPA:
    def __init__(
        self,
        openai : ChatOpenAI,
        vector_index : QdrantVectorStore,
        reranker : Optional[JinaRerank] = None) :
    
        # initialize the pipeline with the components
        self.openai = openai
        self.vector_index = vector_index
        self.reranker = reranker

        # create runnable branch to extract chat histories
        self._set_extract_chat_histories()

    
    # Extract chat history if exists
    def _set_extract_chat_histories(self):
        prompt_template = """
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
        CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_template(prompt_template)

        self._extract_chat_histories = RunnableBranch(
            # If input includes chat_history, we condense it with the follow-up question
            (
                RunnableLambda(lambda x: bool(x.get("chat_histories"))).with_config(
                    run_name="HasChatHistoryCheck"
                ),  # Condense follow-up question and chat into a standalone_question
                RunnablePassthrough.assign(
                    chat_history=lambda x: self._format_chat_histories(x["chat_histories"])
                )
                | CONDENSE_QUESTION_PROMPT
                | self.openai
                | StrOutputParser(),
            ),
            # Else, we have no chat history, so just pass through the question
            RunnableLambda(lambda x : x["question"]),
        )


    # Chat history fromatter
    def _format_chat_histories(self, chat_histories: List[Dict]) -> List:
        print("its here")
        buffer = []
        for message in chat_histories:
            buffer.append(HumanMessage(message["chat_messages"]["user"]))
            buffer.append(AIMessage(content=message["chat_messages"]["assistant"]))
        return buffer
    
    def _retrieve_context_by_vector(self, question) -> List[Document]:
        question = remove_lucene_chars_cust(question)
        return [el for el in self.vector_index.similarity_search(question, k=4)]

    def _retrieve_context_by_vector_with_score_and_rerank(
        self,
        question: str,
        reranker: JinaRerank,
        k_initial: int = 10,
        k_final: int = 3,
        relevance_threshold: float = 0.5,) -> List[Document]:

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
        initial_results = self.vector_index.similarity_search_with_score(cleaned_question, k=k_initial)
        
        # Saring dokumen berdasarkan ambang skor
        filtered_docs = [doc for (doc, score) in initial_results if score >= relevance_threshold]
        
        # Bungkus fungsi penyedia dokumen dalam RunnableLambda (sesuai ekspektasi ContextualCompressionRetriever)
        base_retriever_runnable = RunnableLambda(lambda q: filtered_docs)
        
        # Buat reranker dengan model JinaRerank
        self.reranker = reranker
        
        # Buat pipeline retriever dengan reranking
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.reranker,
            base_retriever=base_retriever_runnable
        )
        
        # Lakukan reranking
        reranked_docs = compression_retriever.invoke(cleaned_question)
        
        # Ambil hanya k_final dokumen teratas
        final_docs = reranked_docs[:k_final]
        
        return final_docs
    
    def _retriever(self, question: str) -> str:
        print(question)
        unstructured_data = self._retrieve_context_by_vector(question)

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

    def get_response(self, question : str, chat_histories : Optional[List[Dict]] = None) :
        prompt_template = """
Your name is OPA. You are a great, friendly and professional AI chat bot about product from the "Central of Oil Palm Research".

### User Question:
{question}

### Context:
{context}

### Important Instructions:
- Base your response only on the provided context. If the contexts provided do not match or don't exist, say you don't know.
- When answering questions, do not include a greeting or introduction unless explicitly requested.

Your Answer:
"""
        PROMPT = ChatPromptTemplate.from_template(prompt_template)

        chain = (
            RunnableParallel(
                {
                    "context": self._extract_chat_histories | self._retriever,
                    "question": RunnablePassthrough(),
                }
            )
            | PROMPT
            | self.openai
            | StrOutputParser()
        )

        return chain.stream(
            {
                "chat_histories" : chat_histories,
                "question" : question
            }
        )
