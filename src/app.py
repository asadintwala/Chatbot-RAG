import streamlit as st # For building web app
from llm.gemini_operations import GeminiChat # Custom GeminiChat class for AI responses
from database.pinecone_operations import PineconeManager # Pinecone manager for storing/retrieving embeddings
from utils.pdf_processor import PDFProcessor
from utils.chat_memory import ChatMemory # Chat memory for storing chat history

# Initializes Streamlit's session state to persist chatbot-related objects.
def initialize_session_state():
    # Initialize chat memory if not already stored in session state
    if 'chat_memory' not in st.session_state:
        st.session_state.chat_memory = ChatMemory()
    # Initialize Gemini chat model if not already stored
    if 'gemini_chat' not in st.session_state:
        st.session_state.gemini_chat = GeminiChat()
    # Initialize Pinecone manager for embedding storage/retrieval
    if 'pinecone_manager' not in st.session_state:
        st.session_state.pinecone_manager = PineconeManager()
# Main function to run the Streamlit chatbot application.
def main():
    st.title("Gemini Chatbot with PDF Knowledge")
    initialize_session_state() # Ensure all necessary components are initialized

    # PDF Upload Section
    st.sidebar.header("Upload PDF") # Sidebar section for file upload
    pdf_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
    
    if pdf_file and st.sidebar.button("Process PDF"):
        """
        If the user uploads a PDF and clicks the "Process PDF" button, 
        extract text, chunk it, and store embeddings in Pinecone.
        """
        with st.spinner("Processing PDF..."): # Show a loading spinner
            # Extract and process PDF
            pdf_processor = PDFProcessor() # Instantiate the PDF processor
            text = pdf_processor.extract_text(pdf_file) # Extract text from the uploaded PDF
            chunks = pdf_processor.create_chunks(text) # Split text into smaller chunks
            
            # Store extracted chunks as embeddings in Pinecone with metadata
            st.session_state.pinecone_manager.store_embeddings(
                chunks, 
                metadata={"source": pdf_file.name} # Store PDF filename as metadata
            )
            st.sidebar.success("PDF processed and stored successfully!")

    # Chat Interface
    st.write("Chat History:") # Display chat history section
    # Loop through and display past messages
    for message in st.session_state.chat_memory.get_history():
        with st.chat_message(message["role"]): # Display message as user or assistant
            st.write(message["content"])

    user_input = st.chat_input("Type your message here...") # Input box for user messages
    
    if user_input:
        """
        When the user sends a message:
        - Display it in the chat
        - Retrieve relevant information from Pinecone
        - Get a response from Gemini AI
        - Display and store the AI response
        """
        with st.chat_message("user"):
            st.write(user_input)
        # Add user message to chat history
        st.session_state.chat_memory.add_message("user", user_input)

        # Retrieve relevant stored chunks from Pinecone for context
        relevant_chunks = st.session_state.pinecone_manager.similarity_search(user_input)
        context = "\n".join(relevant_chunks) if relevant_chunks else None # Combine retrieved chunks into context

        # Get AI-generated response from Gemini model using the retrieved context
        response = st.session_state.gemini_chat.get_response(user_input, context)

        # Display assistant response
        with st.chat_message("assistant"):
            st.write(response)
        # Add AI response to chat history
        st.session_state.chat_memory.add_message("assistant", response)

# Ensure the script runs as the main module
if __name__ == "__main__":
    main() 