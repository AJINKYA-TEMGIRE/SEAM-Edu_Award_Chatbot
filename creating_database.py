from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

emb = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")


def create_db() -> str:
    flag = False

    if flag == False:
        documents = PyPDFLoader("D:\Agentic AI Projects\Law Chatbot\Data\LAW .pdf").load()

        chunks = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150).split_documents(documents)
        for d in chunks:
            d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")

        vectordatabase = FAISS.from_documents(chunks , emb)
        vectordatabase.save_local("faiss_index_database")
    return "Created FAISS Index"

print(create_db())