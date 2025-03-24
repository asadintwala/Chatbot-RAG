from PyPDF2 import PdfReader # For reading pdf files
from config import CHUNK_SIZE, CHUNK_OVERLAP # Config variables for text chunking

class PDFProcessor: #A class to extract text from PDF files and split it into manageable chunks.
    @staticmethod
    def extract_text(pdf_file):
        pdf_reader = PdfReader(pdf_file) # Initialize PDF reader
        text = "" # Initialize an empty string to store extracted text
        # Loop through each page and extract text
        for page in pdf_reader.pages:
            text += page.extract_text() # Append extracted text to the string
        return text # Return the extracted text

    @staticmethod
    def create_chunks(text):
        """
        Splits extracted text into smaller chunks for processing.
        
        :param text: The text to be split
        :return: List of text chunks
        """
        chunks = [] # Initializng empty List to store text chunks
        start = 0 # Starting index for slicing
        # Iterate over text and create overlapping chunks
        while start < len(text):
            end = start + CHUNK_SIZE # Define the chunk endpoint
            # If the chunk endpoint exceeds the text length, take the remaining text
            if end > len(text):
                chunks.append(text[start:])
            else:
                chunks.append(text[start:end]) # Append chunk to the list
            start = end - CHUNK_OVERLAP # Move start index backward to create an overlap
        return chunks  # Return the list of text chunks