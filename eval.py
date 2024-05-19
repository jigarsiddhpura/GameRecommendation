from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from llama_index.llms.anthropic import Anthropic
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_anthropic import AnthropicLLM
from dotenv import load_dotenv
import torch
import os
import time
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

# Load knowledge base
loader = CSVLoader('./data/games.csv', encoding='utf-8')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5120, chunk_overlap=20)
documents = loader.load_and_split(text_splitter)

generator_llm = AnthropicLLM(model="claude-2.1")
# critic_llm = AnthropicLLM(model="claude-3-opus-20240229")  # not working
critic_llm = Anthropic(model="claude-3-opus-20240229")
ollama_emb = HuggingFaceEmbeddings(model_name="Snowflake/snowflake-arctic-embed-m")

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    ollama_emb
)

batch_size = 6
batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]

for i,batch in enumerate(batches):
    testset_batch = generator.generate_with_langchain_docs(
        batch,
        test_size=10,
        distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
        raise_exceptions=False
    )
    df_batch = testset_batch.to_pandas()
    df_batch.to_csv(f'./batch_{i}.csv', index=False)
    time.sleep(50)  # Sleep for 50 seconds after each batch to prevent rate_limit errors
    print(f"Batch {i} completed.")

