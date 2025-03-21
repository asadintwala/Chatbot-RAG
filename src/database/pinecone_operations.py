from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

class PineconeManager:
    def __init__(self):
        pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        #if PINECONE_INDEX_NAME not in pc.list_indexes():
        #    pc.create_index(PINECONE_INDEX_NAME, dimension=1024, metric="cosine")
        self.index = pc.Index(PINECONE_INDEX_NAME)
        self.embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')

    def store_embeddings(self, text_chunks, metadata=None):
        vectors = []
        
        for i, chunk in enumerate(text_chunks):
            embedding = self.embedding_model.encode(chunk, normalize_embeddings=True)
            embedding = embedding.tolist()  # Convert numpy array to list
            
            vector = {
                'id': f'chunk_{i}',
                'values': embedding,
                'metadata': {'text': chunk, **(metadata or {})}
            }
            vectors.append(vector)
        
        self.index.upsert(vectors=vectors)

    def similarity_search(self, query, top_k=3):
        query_embedding = self.embedding_model.encode(query, normalize_embeddings=True)
        query_embedding = query_embedding.tolist()  # Convert numpy array to list
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return [match.metadata['text'] for match in results.matches] 