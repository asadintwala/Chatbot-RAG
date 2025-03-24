import streamlit as st
from llm.gemini_operations import GeminiChat
from database.pinecone_operations import PineconeManager
from utils.pdf_processor import PDFProcessor
from utils.chat_memory import ChatMemory

def initialize_session_state():
    if 'chat_memory' not in st.session_state:
        st.session_state.chat_memory = ChatMemory()
    if 'gemini_chat' not in st.session_state:
        st.session_state.gemini_chat = GeminiChat()
    if 'pinecone_manager' not in st.session_state:
        st.session_state.pinecone_manager = PineconeManager()

def main():
    st.title("Gemini Chatbot with PDF Knowledge")
    initialize_session_state()

    # PDF Upload Section
    st.sidebar.header("Upload PDF")
    pdf_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
    
    if pdf_file and st.sidebar.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            # Extract and process PDF
            pdf_processor = PDFProcessor()
            
            with st.status("Processing document...") as status:
                status.write("Extracting text from PDF...")
                text = pdf_processor.extract_text(pdf_file)
                
                status.write("Creating text chunks...")
                chunks = pdf_processor.create_chunks(text)
                
                status.write(f"Uploading {len(chunks)} chunks to Pinecone...")
                progress_bar = st.progress(0)
                
                # Store in Pinecone with progress tracking
                batch_size = 50
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    st.session_state.pinecone_manager.store_embeddings(
                        batch, 
                        metadata={"source": pdf_file.name}
                    )
                    progress = min((i + batch_size) / len(chunks), 1.0)
                    progress_bar.progress(progress)
                
                status.update(label="PDF processed successfully!", state="complete")

    # Chat Interface
    st.write("Chat History:")
    for message in st.session_state.chat_memory.get_history():
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.chat_memory.add_message("user", user_input)

        # Search for relevant context in Pinecone
        relevant_chunks = st.session_state.pinecone_manager.similarity_search(user_input)
        context = "\n".join(relevant_chunks) if relevant_chunks else None

        # Get response from Gemini
        response = st.session_state.gemini_chat.get_response(user_input, context)

        # Display assistant response
        with st.chat_message("assistant"):
            st.write(response)
        st.session_state.chat_memory.add_message("assistant", response)

if __name__ == "__main__":
    main() 
