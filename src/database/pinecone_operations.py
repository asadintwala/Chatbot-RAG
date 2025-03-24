from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME
from sentence_transformers import SentenceTransformer # for generating embedding from text

class PineconeManager:
    """Class to manage interactions with Pinecone for storing and searching embeddings."""
    def __init__(self):
        """Initialize the Pinecone client, connect to an existing index, and load the embedding model."""
        pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        self.index = pc.Index(PINECONE_INDEX_NAME) # Conencting to existing pc index
        # Load the sentence transformer model to generate embeddings
        self.embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')

    def store_embeddings(self, text_chunks, metadata=None):
        """Stores embeddings of given text chunks in Pinecone."""
        vectors = [] # intilaizing empty List to hold all vector entries
        # Iterate over the text chunks and generate embeddings
        for i, chunk in enumerate(text_chunks):
            # Generate embedding for the current text chunk and normalize it
            embedding = self.embedding_model.encode(chunk, normalize_embeddings=True)
            embedding = embedding.tolist()  # Convert numpy array to list
            # Create a dictionary representing the vector entry
            vector = {
                'id': f'chunk_{i}', # Unique ID for each chunk
                'values': embedding, # The generated embedding
                'metadata': {'text': chunk, **(metadata or {})} # Store original text and any additional metadata
            }
            vectors.append(vector) # Append to list of vectors
        # Upload the generated embeddings to Pinecone
        self.index.upsert(vectors=vectors)

    def similarity_search(self, query, top_k=3):
        # Performs similarity search in Pinecone for a given query.
        # :param query: The input query string to search for
        # :param top_k: The number of top similar results to retrieve (default: 3)

        # Generate embedding for the query and normalize it
        query_embedding = self.embedding_model.encode(query, normalize_embeddings=True)
        query_embedding = query_embedding.tolist()  # Convert numpy array to list
        # Perform similarity search in Pinecone
        results = self.index.query(
            vector=query_embedding, # Query vector
            top_k=top_k, # Number of top matches to return
            include_metadata=True # Include metadata in the search results
        )
        # Extract and return the text of the matched results
        return [match.metadata['text'] for match in results.matches] 