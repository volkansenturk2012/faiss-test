from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import PagedCSVReader
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import VectorStoreIndex
import faiss
import os
import pandas as pd
from dotenv import load_dotenv


# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Llamaindex global settings for llm and embeddings
EMBED_DIMENSION=512
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=EMBED_DIMENSION)

file_path = ('data/clean-202412R3.csv') # insert the path of the csv file

data = pd.read_csv(file_path)

# Preview the csv file
data.head()

# Create FaisVectorStore to store embeddings
fais_index = faiss.IndexFlatL2(EMBED_DIMENSION)
vector_store = FaissVectorStore(faiss_index=fais_index)