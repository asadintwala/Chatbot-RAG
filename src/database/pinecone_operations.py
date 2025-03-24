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
        self.batch_size = 50  # Adjust this number based on your average vector size

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
            
            # Upload in batches to avoid size limit
            if len(vectors) >= self.batch_size:
                self._upsert_batch(vectors)
                vectors = []  # Clear the batch
        
        # Upload any remaining vectors
        if vectors:
            self._upsert_batch(vectors)

    def _upsert_batch(self, vectors):
        try:
            self.index.upsert(vectors=vectors)
        except Exception as e:
            print(f"Error upserting batch: {str(e)}")
            # If batch is still too large, reduce batch size and retry
            if len(vectors) > 1:
                mid = len(vectors) // 2
                self._upsert_batch(vectors[:mid])
                self._upsert_batch(vectors[mid:])

    def similarity_search(self, query, top_k=3):
        query_embedding = self.embedding_model.encode(query, normalize_embeddings=True)
        query_embedding = query_embedding.tolist()  # Convert numpy array to list
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return [match.metadata['text'] for match in results.matches] 
