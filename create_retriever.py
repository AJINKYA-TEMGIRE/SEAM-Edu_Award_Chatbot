from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

emb = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")


def r():
    database = FAISS.load_local("faiss_index_database" , emb , allow_dangerous_deserialization=True)
    retriever = database.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k" : 5}
    )
    return retriever