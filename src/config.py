import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

# Pinecone Settings
PINECONE_INDEX_NAME = "chatbot"
# AI gets better context when answering questions, especially for long documents.
CHUNK_SIZE = 1000 # No. of characters in each chunk of text
CHUNK_OVERLAP = 200 # The next chunk will start from 800-1800 characters to maintain context