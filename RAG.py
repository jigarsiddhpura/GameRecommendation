import chromadb
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate, StorageContext, load_index_from_storage
from llama_index.llms.anthropic import Anthropic
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from functions import delta_index
import os
load_dotenv()

PERSIST_DIR = "./vectorDB"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

# global default
embed_model = HuggingFaceEmbedding(model_name = "Snowflake/snowflake-arctic-embed-m")
Settings.embed_model = embed_model
Settings.llm = Anthropic(model="claude-3-opus-20240229")

## Loading the knowledge base
# if not (os.path.join(PERSIST_DIR)):
documents :list = SimpleDirectoryReader("./data").load_data()

# pipeline = IngestionPipeline(transformations=[SentenceSplitter()])
# nodes = pipeline.run(documents=documents)

## Indexing and storing

# delta indexing
# unique_ids, unique_documents = delta_index(documents)

# initialize client & collection, setting path to save data
db = chromadb.PersistentClient(path=PERSIST_DIR)
chroma_collection = db.get_or_create_collection("games")

# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

if not (os.path.join(PERSIST_DIR,'index')):
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
else:
    print("loading from storage")
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=f"{PERSIST_DIR}/index")
    vector_index = load_index_from_storage(storage_context)

# create your index if doesn't exist

if not (os.path.join(PERSIST_DIR,'index')):
    print("creating index")
    vector_index.storage_context.persist(persist_dir=f"{PERSIST_DIR}/index")

Settings.vector_index = vector_index    

tokenizer = Anthropic().tokenizer
Settings.tokenizer = tokenizer

## query_engine
query_engine = vector_index.as_query_engine(similarity_top_k = 4)

# shakespeare!
qa_prompt_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query. In case you don't know the answer say\n"
    "'I dont't know!' \n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
)

response = query_engine.query("I want to play shooting games today, do you have any recommendations for me?")
print(type(response))
print(str(response))