import streamlit as st
import time
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate, StorageContext, load_index_from_storage
from llama_index.llms.anthropic import Anthropic
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_community.document_loaders.csv_loader import CSVLoader
from llama_index.core import SimpleDirectoryReader
# from functions import delta_index
import chromadb

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

if "key" not in st.session_state:
    st.session_state.key = ANTHROPIC_API_KEY
os.environ["ANTHROPIC_API_KEY"] = st.session_state.key

PERSIST_DIR = "./vectorDB"

st.subheader("PLAY EPIC 🤖 ")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "unique_ids" not in st.session_state:
    st.session_state.unique_ids = []

@st.cache_resource
def load_model():
    embed_model = HuggingFaceEmbedding(model_name="Snowflake/snowflake-arctic-embed-m")
    Settings.embed_model = embed_model
    Settings.llm = Anthropic(model="claude-3-opus-20240229")

    documents = SimpleDirectoryReader("./data").load_data()
    # pipeline = IngestionPipeline(transformations=[SentenceSplitter()])
    # nodes = pipeline.run(documents=documents)
    
    # unique_ids, unique_documents = delta_index(documents)

    db = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = db.get_or_create_collection("games")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
    if not (os.path.join(PERSIST_DIR,'index')):
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        # st.session_state.unique_ids = list(set(unique_ids + st.session_state.unique_ids))
    else:
        print("loading from storage")
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=f"{PERSIST_DIR}/index")
        vector_index = load_index_from_storage(storage_context)

    if not (os.path.join(PERSIST_DIR,'index')):
        print("creating index")
        vector_index.storage_context.persist(persist_dir=f"{PERSIST_DIR}/index")

    Settings.vector_index = vector_index

    tokenizer = Anthropic().tokenizer
    Settings.tokenizer = tokenizer

    qa_prompt_tmpl_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the query as a best game recommender. In case you don't know the answer say\n"
        "'I dont't know!' \n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

    query_engine = vector_index.as_query_engine(similarity_top_k=4)
    query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})

    return query_engine

query_engine = load_model()

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if prompt := st.chat_input("What's the play mood?!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Collecting answer"):
            message_placeholder = st.empty()
            full_response = ""
            result = query_engine.query(prompt)
            assistant_response = str(result)

    for chunk in assistant_response:
        full_response += chunk + ""
        time.sleep(0.01)
        message_placeholder.markdown(full_response + "▌")

    message_placeholder.markdown(full_response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })