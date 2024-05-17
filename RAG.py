import chromadb
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate, StorageContext
from llama_index.llms.anthropic import Anthropic
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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
documents = SimpleDirectoryReader("./data").load_data()

## Indexing and storing

# initialize client, setting path to save data
db = chromadb.PersistentClient(path=PERSIST_DIR)

chroma_collection = db.get_or_create_collection("games")

# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# create your index
vector_index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
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