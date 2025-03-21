from PyPDF2 import PdfReader
from config import CHUNK_SIZE, CHUNK_OVERLAP

class PDFProcessor:
    @staticmethod
    def extract_text(pdf_file):
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    @staticmethod
    def create_chunks(text):
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            if end > len(text):
                chunks.append(text[start:])
            else:
                chunks.append(text[start:end])
            start = end - CHUNK_OVERLAP
        return chunks 