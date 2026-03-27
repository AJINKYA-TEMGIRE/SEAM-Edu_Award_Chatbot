from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
import os

#  Embedding model (fast + good quality)
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Chunking (important for retrieval quality)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120
)

def create_db():
    pdf_files = [
        r"D:\Agentic AI Projects\Law Chatbot\Data\Adams and Victor's Principles of Neurology 11th Edition.pdf",
        r"D:\Agentic AI Projects\Law Chatbot\Data\Medical.pdf",
        r"D:\Agentic AI Projects\Law Chatbot\Data\Oxford Textbook of Clinical Nephrology-4th edition (1).pdf",
        r"D:\Agentic AI Projects\Law Chatbot\Data\Sleisenger 11th edition.pdf",
        r"D:\Agentic AI Projects\Law Chatbot\Data\Wintrobe's Clinical Hematology 14e 2019.pdf",
        r"D:\Agentic AI Projects\Law Chatbot\Data\Braunwalds Heart Disease 12th ed.pdf"
    ]

    db = None

    for file in pdf_files:
        print(f"Processing: {file}")

        loader = PyMuPDFLoader(file)
        documents = loader.load()

        chunks = splitter.split_documents(documents)

        # Clean text
        for d in chunks:
            d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")

        # Incremental FAISS build (VERY IMPORTANT for 8GB RAM)
        if db is None:
            db = FAISS.from_documents(chunks, emb)
        else:
            db.add_documents(chunks)

    #  Save database
    db.save_local("faiss_index_database")

    return "FAISS database created successfully"


if __name__ == "__main__":
    print(create_db())