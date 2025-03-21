from setuptools import setup, find_packages

setup(
    name="gemini-chatbot",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "google-generativeai",
        "python-dotenv",
        "pinecone-client",
        "PyPDF2",
        "langchain",
    ],
) 